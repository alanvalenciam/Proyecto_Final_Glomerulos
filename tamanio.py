import os
import math
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from multiprocessing import Pool, cpu_count
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Utilidades GeoJSON y búsqueda de pares
# =========================


def load_geojson(geojson_path: str) -> Dict:
    """Carga un archivo GeoJSON y retorna el objeto."""
    with open(geojson_path, 'r') as f:
        return json.load(f)


def get_geojson_bounds(geojson_obj: Dict) -> Tuple[float, float, float, float]:
    """
    Extrae el bbox del GeoJSON (minx, miny, maxx, maxy).
    Si no hay bbox, calcula desde las geometrías.
    """
    if "bbox" in geojson_obj:
        bbox = geojson_obj["bbox"]
        return tuple(bbox[:4])

    # Calcular bbox desde features
    all_coords = []
    for feature in geojson_obj.get("features", []):
        geom = feature.get("geometry", {})
        if geom:
            try:
                shp = shape(geom)
                bounds = shp.bounds
                all_coords.append(bounds)
            except:
                pass

    if all_coords:
        minx = min(b[0] for b in all_coords)
        miny = min(b[1] for b in all_coords)
        maxx = max(b[2] for b in all_coords)
        maxy = max(b[3] for b in all_coords)
        return (minx, miny, maxx, maxy)

    return (0, 0, 0, 0)


def find_geojson_files(input_dir: str) -> List[str]:
    """
    Busca automáticamente en input_dir todos los archivos .geojson.
    Retorna lista ordenada de paths absolutos.
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.warning(f"Directory does not exist: {input_dir}")
        return []

    files = sorted(str(p) for p in input_path.glob("*.geojson"))
    if not files:
        logger.warning(f"No .geojson files found in {input_dir}")
    return files


def _safe_worker_count(geojson_paths: List[str]) -> int:
    """
    Determine safe number of parallel workers based on available RAM.

    Strategy:
    - Get available RAM and reserve 30% (safety margin)
    - Estimate per-file RAM: max_file_size × 2 (GeoJSON parsing overhead)
    - workers = min(cpu_count(), max(1, available_ram / max_file_ram))

    Returns:
        int: Number of workers to use in the pool (minimum 1)
    """
    try:
        if PSUTIL_AVAILABLE:
            available_ram = psutil.virtual_memory().available
        else:
            available_ram = 4 * 1024 * 1024 * 1024  # Conservative fallback: 4GB

        safe_ram = available_ram * 0.7

        max_file_size = max(
            os.path.getsize(p) for p in geojson_paths
        ) if geojson_paths else 10 * 1024 * 1024

        estimated_per_file_ram = max_file_size * 2

        workers_by_ram = max(1, int(safe_ram / estimated_per_file_ram))
        workers = min(cpu_count(), workers_by_ram)

        logger.info(f"RAM-aware worker calculation: available={safe_ram / 1e9:.1f}GB, max_file={max_file_size / 1e6:.1f}MB, workers={workers}")
        return workers

    except Exception as e:
        logger.warning(f"psutil RAM calculation failed ({e}), using conservative worker count")
        return min(4, cpu_count())


def _process_single_geojson(args) -> Tuple[str, List[Dict]]:
    """
    Worker function to analyze a single GeoJSON file in parallel.

    Returns:
        (geojson_path, rows_list) where rows_list is the processed geometries
    """
    geojson_path, zoom_scale = args

    rows = []
    dataset_name = Path(geojson_path).stem

    try:
        geojson_obj = load_geojson(geojson_path)
        features = geojson_obj.get("features", [])

        if not features:
            logger.info(f"  SKIP - {dataset_name}: sin geometrías")
            return (geojson_path, rows)

        for idx, feature in enumerate(features):
            geom_dict = feature.get("geometry", {})
            props = feature.get("properties", {})

            if not geom_dict:
                continue

            try:
                geom = shape(geom_dict)
                area = geom.area
                minx, miny, maxx, maxy = geom.bounds
                width = maxx - minx
                height = maxy - miny
                square_size = max(width, height)
                perimeter = geom.length

                if perimeter > 0:
                    circularity = (4.0 * math.pi * area) / (perimeter ** 2)
                else:
                    circularity = 0.0

                glomeruli_name = props.get("name", f"glomeruli_{idx+1}")

                rows.append({
                    "dataset": dataset_name,
                    "name": glomeruli_name,
                    "area_native_px2": round(area, 2),
                    "width_native_px": round(width, 2),
                    "height_native_px": round(height, 2),
                    "square_size_native_px": round(square_size, 2),
                    "perimeter_native_px": round(perimeter, 2),
                    "circularity": round(circularity, 4),
                    "area_tile_px2": round(area * zoom_scale ** 2, 2),
                    "width_tile_px": round(width * zoom_scale, 2),
                    "height_tile_px": round(height * zoom_scale, 2),
                    "square_size_tile_px": round(square_size * zoom_scale, 2),
                })
            except Exception as e:
                logger.warning(f"Error en {dataset_name} feature {idx}: {e}")

    except Exception as e:
        logger.error(f"Error procesando {dataset_name}: {e}")

    return (geojson_path, rows)




# ============================
# Análisis de tamaños de glomérulos
# ============================


# ============================
# Funciones helper para análisis de tamaño de tile
# ============================


def calculate_tile_percentiles(square_sizes: List[float]) -> Dict[str, float]:
    """
    Calcula percentiles del tamaño cuadrado.
    Retorna dict con P50, P75, P85, P90, P95.
    """
    return {
        "p50": float(np.percentile(square_sizes, 50)),
        "p75": float(np.percentile(square_sizes, 75)),
        "p85": float(np.percentile(square_sizes, 85)),
        "p90": float(np.percentile(square_sizes, 90)),
        "p95": float(np.percentile(square_sizes, 95)),
    }


def recommend_tile_size(percentil_95: float) -> int:
    """
    Redondea percentil 95% al siguiente múltiplo de 128 o 256.
    Retorna tamaño recomendado en píxeles.
    """
    # Opciones de múltiplos estándar
    candidates = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2176, 2304, 2432, 2560]

    # Encuentra el primero >= percentil_95
    for size in candidates:
        if size >= percentil_95:
            return size

    # Si excede todos, retorna el mayor
    return candidates[-1]


def calculate_coverage_percentage(square_sizes: List[float], tile_size: int) -> float:
    """
    Calcula qué % de glomérulos cabrían completamente en un tile_size × tile_size.
    square_sizes: lista de tamaños cuadrados
    tile_size: tamaño del tile en píxeles
    Retorna: % de glomérulos que caben (square_size <= tile_size)
    """
    if not square_sizes:
        return 0.0
    count_fit = sum(1 for s in square_sizes if s <= tile_size)
    return 100.0 * count_fit / len(square_sizes)


def analyze_glomeruli_sizes(geojson_path: str,
                           zoom_scale: float = 0.5,
                           output_dir: str = "analysis_plots",
                           output_csv: str = "glomeruli_sizes.csv") -> Tuple[pd.DataFrame, Dict]:
    """
    Mide el tamaño de cada glomérulo en un GeoJSON.
    Convierte a tile-space usando zoom_scale.
    Calcula tamaño de tile cuadrado recomendado basado en percentil 95%.

    Args:
        geojson_path: ruta al archivo GeoJSON
        zoom_scale: factor de escala (1 tile px = native px × zoom_scale, default 0.5)
        output_dir: directorio para guardar plots
        output_csv: ruta para guardar CSV

    Retorna:
    - DataFrame: una fila por glomérulo (columnas nativas y tile-space)
    - Dict: estadísticas agregadas en tile-space
    """
    os.makedirs(output_dir, exist_ok=True)

    # Cargar GeoJSON
    geojson_obj = load_geojson(geojson_path)
    features = geojson_obj.get("features", [])

    if not features:
        logger.warning(f"No hay geometrías en {geojson_path}")
        return pd.DataFrame(), {}

    rows = []

    # Procesar cada polígono
    for idx, feature in enumerate(features):
        geom_dict = feature.get("geometry", {})
        props = feature.get("properties", {})

        if not geom_dict:
            continue

        try:
            geom = shape(geom_dict)

            # Calcular área (en píxeles²)
            area = geom.area

            # Calcular bounding box
            minx, miny, maxx, maxy = geom.bounds
            width = maxx - minx
            height = maxy - miny

            # Calcular tamaño cuadrado (max(ancho, alto))
            square_size = max(width, height)

            # Calcular perímetro (en píxeles)
            perimeter = geom.length

            # Calcular circularidad = 4π × área / perímetro²
            # 1.0 = círculo perfecto, < 1.0 = más irregular
            if perimeter > 0:
                circularity = (4.0 * math.pi * area) / (perimeter ** 2)
            else:
                circularity = 0.0

            # Nombre del glomérulo
            glomeruli_name = props.get("name", f"glomeruli_{idx+1}")

            # Calcular valores en tile-space (lo que el modelo ve)
            area_tile = area * (zoom_scale ** 2)
            width_tile = width * zoom_scale
            height_tile = height * zoom_scale
            square_size_tile = square_size * zoom_scale

            rows.append({
                "name": glomeruli_name,
                "area_native_px2": round(area, 2),
                "width_native_px": round(width, 2),
                "height_native_px": round(height, 2),
                "square_size_native_px": round(square_size, 2),
                "perimeter_native_px": round(perimeter, 2),
                "circularity": round(circularity, 4),
                "area_tile_px2": round(area_tile, 2),
                "width_tile_px": round(width_tile, 2),
                "height_tile_px": round(height_tile, 2),
                "square_size_tile_px": round(square_size_tile, 2),
                "feature_index": idx
            })

        except Exception as e:
            logger.warning(f"Error procesando feature {idx}: {e}")

    if not rows:
        logger.error("No se pudieron procesar geometrías válidas.")
        return pd.DataFrame(), {}

    # Crear DataFrame
    df = pd.DataFrame(rows)

    # Calcular estadísticas base (en tile-space)
    stats = {
        "zoom_scale": zoom_scale,
        "count": len(df),
        "area": {
            "min": float(df["area_tile_px2"].min()),
            "max": float(df["area_tile_px2"].max()),
            "mean": float(df["area_tile_px2"].mean()),
            "std": float(df["area_tile_px2"].std())
        },
        "width": {
            "min": float(df["width_tile_px"].min()),
            "max": float(df["width_tile_px"].max()),
            "mean": float(df["width_tile_px"].mean()),
            "std": float(df["width_tile_px"].std())
        },
        "height": {
            "min": float(df["height_tile_px"].min()),
            "max": float(df["height_tile_px"].max()),
            "mean": float(df["height_tile_px"].mean()),
            "std": float(df["height_tile_px"].std())
        },
        "square_size_native": {
            "min": float(df["square_size_native_px"].min()),
            "max": float(df["square_size_native_px"].max()),
            "mean": float(df["square_size_native_px"].mean()),
            "std": float(df["square_size_native_px"].std())
        },
        "square_size_tile": {
            "min": float(df["square_size_tile_px"].min()),
            "max": float(df["square_size_tile_px"].max()),
            "mean": float(df["square_size_tile_px"].mean()),
            "std": float(df["square_size_tile_px"].std())
        },
        "perimeter": {
            "min": float(df["perimeter_native_px"].min()),
            "max": float(df["perimeter_native_px"].max()),
            "mean": float(df["perimeter_native_px"].mean()),
            "std": float(df["perimeter_native_px"].std())
        },
        "circularity": {
            "min": float(df["circularity"].min()),
            "max": float(df["circularity"].max()),
            "mean": float(df["circularity"].mean())
        }
    }

    # Calcular percentiles de tamaño cuadrado (en tile-space)
    percentiles = calculate_tile_percentiles(df["square_size_tile_px"].tolist())
    stats["percentiles"] = percentiles

    # Calcular tamaño de tile recomendado (basado en tile-space)
    p95_size = percentiles["p95"]
    recommended_size = recommend_tile_size(p95_size)
    coverage_pct = calculate_coverage_percentage(df["square_size_tile_px"].tolist(), recommended_size)

    stats["tile_recommendation"] = {
        "p95_raw": p95_size,
        "recommended_size": recommended_size,
        "coverage_percentage": coverage_pct,
        "glomeruli_covered": int(np.ceil(len(df) * coverage_pct / 100.0)),
        "glomeruli_incomplete": len(df) - int(np.ceil(len(df) * coverage_pct / 100.0))
    }

    # Guardar CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"DataFrame de glomérulos guardado en {output_csv}")

    # Crear visualizaciones
    _create_glomeruli_plots(df, stats, output_dir)

    # Generar reporte de texto
    dataset_name = Path(geojson_path).stem
    _generate_tile_report(dataset_name, df, stats, output_dir)

    # Imprimir resumen
    _print_glomeruli_summary(dataset_name, stats)

    return df, stats


def _create_glomeruli_plots(df: pd.DataFrame,
                           stats: Dict,
                           output_dir: str):
    """Crea 4 gráficos: histograma de áreas, scatter width vs height, histograma de circularidad, histograma de tamaño cuadrado."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Histograma de áreas (tile-space)
    plt.figure(figsize=(10, 6))
    plt.hist(df["area_tile_px2"], bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Area (tile-space pixels²)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title("Distribution of Glomeruli Areas", fontsize=13, fontweight="bold")
    plt.axvline(stats["area"]["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean: {stats['area']['mean']:.0f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    area_plot = os.path.join(output_dir, "01_histogram_areas.png")
    plt.savefig(area_plot, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot guardado: {area_plot}")

    # 2. Scatter: width vs height (tile-space)
    plt.figure(figsize=(10, 6))
    plt.scatter(df["width_tile_px"], df["height_tile_px"], s=100, alpha=0.6, color="darkgreen", edgecolors="black")
    plt.xlabel("Width (tile-space pixels)", fontsize=11)
    plt.ylabel("Height (tile-space pixels)", fontsize=11)
    plt.title("Glomeruli Dimensions: Width vs Height", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    # Añade línea de tendencia si hay variabilidad
    z = np.polyfit(df["width_tile_px"], df["height_tile_px"], 1)
    p = np.poly1d(z)
    plt.plot(df["width_tile_px"], p(df["width_tile_px"]), "r--", linewidth=2, label="Trend")
    plt.legend()
    scatter_plot = os.path.join(output_dir, "02_scatter_dimensions.png")
    plt.savefig(scatter_plot, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot guardado: {scatter_plot}")

    # 3. Histograma de circularidad
    plt.figure(figsize=(10, 6))
    plt.hist(df["circularity"], bins=15, color="coral", edgecolor="black", alpha=0.7)
    plt.xlabel("Circularity", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title("Distribution of Glomeruli Circularity (1.0 = perfect circle)", fontsize=13, fontweight="bold")
    plt.axvline(stats["circularity"]["mean"], color="blue", linestyle="--", linewidth=2, label=f"Mean: {stats['circularity']['mean']:.3f}")
    plt.axvline(1.0, color="green", linestyle=":", linewidth=2, label="Perfect circle")
    plt.legend()
    plt.grid(True, alpha=0.3)
    circ_plot = os.path.join(output_dir, "03_histogram_circularity.png")
    plt.savefig(circ_plot, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot guardado: {circ_plot}")

    # 4. Histograma de tamaño cuadrado con percentiles (tile-space)
    plt.figure(figsize=(12, 7))
    square_sizes = df["square_size_tile_px"].tolist()
    bins = max(15, len(set(square_sizes)) // 2)
    plt.hist(square_sizes, bins=bins, color="skyblue", edgecolor="black", alpha=0.7, label="Distribution")

    # Línea de media (azul)
    mean_size = stats["square_size_tile"]["mean"]
    plt.axvline(mean_size, color="blue", linestyle="-", linewidth=2.5, label=f"Mean: {mean_size:.0f} px")

    # Línea de P95 (rojo, discontinua)
    p95_size = stats["percentiles"]["p95"]
    plt.axvline(p95_size, color="red", linestyle="--", linewidth=2.5, label=f"P95: {p95_size:.0f} px")

    # Línea de tamaño recomendado (verde, punteada)
    rec_size = stats["tile_recommendation"]["recommended_size"]
    plt.axvline(rec_size, color="green", linestyle=":", linewidth=2.5, label=f"Recommended: {rec_size} px")

    # Anotaciones
    y_max = plt.ylim()[1]
    plt.text(mean_size, y_max * 0.95, f"{mean_size:.0f}", ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    plt.text(p95_size, y_max * 0.90, f"{p95_size:.0f}", ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    plt.text(rec_size, y_max * 0.85, f"{rec_size}", ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    plt.xlabel("Square Tile Size (tile-space pixels)", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency (number of glomeruli)", fontsize=12, fontweight="bold")
    plt.title("Distribution of Square Tile Sizes for Glomeruli Coverage", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, alpha=0.3)

    square_plot = os.path.join(output_dir, "04_histogram_square_sizes.png")
    plt.savefig(square_plot, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot guardado: {square_plot}")


def _generate_tile_report(dataset_name: str, df: pd.DataFrame, stats: Dict, output_dir: str):
    """Genera un archivo de reporte de texto con recomendación de tamaño de tile."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("GLOMERULI SIZE DISTRIBUTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Dataset: {dataset_name}")
    report_lines.append(f"Total glomeruli analyzed: {stats['count']}")
    report_lines.append(f"Zoom scale: {stats['zoom_scale']}  (1 tile-space px = {1/stats['zoom_scale']:.0f} native px)")
    report_lines.append("All sizes reported in TILE SPACE (what the model sees).")
    report_lines.append("")

    # Advertencia si dataset es pequeño
    if stats['count'] < 10:
        report_lines.append("[WARNING] Dataset tiene menos de 10 glomérulos. Las estadísticas pueden no ser representativas.")
        report_lines.append("")

    report_lines.append(f"SQUARE TILE SIZE (tile-space pixels × pixels, zoom_scale={stats['zoom_scale']}):")
    report_lines.append("")
    report_lines.append("Statistics:")
    report_lines.append(f"  Min:        {stats['square_size_tile']['min']:.0f} px")
    report_lines.append(f"  Max:        {stats['square_size_tile']['max']:.0f} px")
    report_lines.append(f"  Mean:       {stats['square_size_tile']['mean']:.0f} px (±{stats['square_size_tile']['std']:.0f})")
    report_lines.append(f"  Median:     {stats['percentiles']['p50']:.0f} px")
    report_lines.append("")
    report_lines.append("Percentiles:")
    report_lines.append(f"  P50 (Median):  {stats['percentiles']['p50']:.0f} px  [50% of glomeruli are ≤{stats['percentiles']['p50']:.0f} px]")
    report_lines.append(f"  P75:           {stats['percentiles']['p75']:.0f} px  [75% of glomeruli are ≤{stats['percentiles']['p75']:.0f} px]")
    report_lines.append(f"  P85:           {stats['percentiles']['p85']:.0f} px  [85% of glomeruli are ≤{stats['percentiles']['p85']:.0f} px]")
    report_lines.append(f"  P90:           {stats['percentiles']['p90']:.0f} px  [90% of glomeruli are ≤{stats['percentiles']['p90']:.0f} px]")
    report_lines.append(f"  P95:           {stats['percentiles']['p95']:.0f} px  [95% of glomeruli are ≤{stats['percentiles']['p95']:.0f} px]")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("RECOMMENDATION")
    report_lines.append("=" * 80)
    report_lines.append("")

    rec = stats["tile_recommendation"]
    report_lines.append(f"To capture {rec['coverage_percentage']:.0f}% of glomeruli completely:")
    report_lines.append("")
    report_lines.append(f"  ✓ Recommended tile size:  {rec['recommended_size']} × {rec['recommended_size']} pixels")
    report_lines.append(f"     (Round {rec['p95_raw']:.0f} px to next multiple of 128)")
    report_lines.append("")
    report_lines.append(f"  ✓ With this size:")
    report_lines.append(f"     - You will capture:  {rec['coverage_percentage']:.0f}% of glomeruli ({rec['glomeruli_covered']}/{stats['count']})")
    report_lines.append(f"     - Incomplete:         {rec['coverage_percentage'] - 100:.0f}% ({rec['glomeruli_incomplete']} glomeruli) [rounded]")
    report_lines.append("")

    # Alternativa más conservadora
    conservative_size = recommend_tile_size(stats['square_size_tile']['max'])
    if conservative_size > rec['recommended_size']:
        conservative_cov = calculate_coverage_percentage(df["square_size_tile_px"].tolist(), conservative_size)
        report_lines.append(f"  Alternative (more conservative):")
        report_lines.append(f"  • Size:  {conservative_size} × {conservative_size} pixels")
        report_lines.append(f"     - You will capture:  {conservative_cov:.0f}% of glomeruli (all {stats['count']})")
        report_lines.append("")

    report_lines.append("=" * 80)

    # Guardar reporte
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "glomeruli_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"Reporte guardado: {report_path}")
    return report_text


def _print_glomeruli_summary(dataset_name: str, stats: Dict):
    """Imprime un resumen formateado de estadísticas, incluyendo recomendación de tile."""
    logger.info("\n" + "=" * 80)
    logger.info("GLOMERULI SIZE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Total glomeruli: {stats['count']}")
    logger.info(f"Zoom scale: {stats['zoom_scale']}  (1 tile-space px = {1/stats['zoom_scale']:.0f} native px)")
    logger.info("")

    # Advertencia si dataset es pequeño
    if stats['count'] < 10:
        logger.warning("Dataset tiene menos de 10 glomérulos. Las estadísticas pueden no ser representativas.")
        logger.info("")

    logger.info("AREA (tile-space pixels²):")
    logger.info(f"  Min: {stats['area']['min']:>12,.0f}  |  Max: {stats['area']['max']:>12,.0f}")
    logger.info(f"  Mean: {stats['area']['mean']:>10,.0f} ± {stats['area']['std']:.0f}")
    logger.info("")

    logger.info("SQUARE TILE SIZE (tile-space pixels):")
    logger.info(f"  Min: {stats['square_size_tile']['min']:>10,.0f}  |  Max: {stats['square_size_tile']['max']:>10,.0f}")
    logger.info(f"  Mean: {stats['square_size_tile']['mean']:>10,.0f} ± {stats['square_size_tile']['std']:.0f}")
    logger.info("")
    logger.info("  Percentiles:")
    logger.info(f"    P50: {stats['percentiles']['p50']:>8,.0f} px  |  P75: {stats['percentiles']['p75']:>8,.0f} px  |  P85: {stats['percentiles']['p85']:>8,.0f} px")
    logger.info(f"    P90: {stats['percentiles']['p90']:>8,.0f} px  |  P95: {stats['percentiles']['p95']:>8,.0f} px")
    logger.info("")

    logger.info("DIMENSIONS (pixels):")
    logger.info(f"  Width:")
    logger.info(f"    Min: {stats['width']['min']:>10,.0f}  |  Max: {stats['width']['max']:>10,.0f}  |  Mean: {stats['width']['mean']:>10,.0f} ± {stats['width']['std']:.0f}")
    logger.info(f"  Height:")
    logger.info(f"    Min: {stats['height']['min']:>10,.0f}  |  Max: {stats['height']['max']:>10,.0f}  |  Mean: {stats['height']['mean']:>10,.0f} ± {stats['height']['std']:.0f}")
    logger.info("")

    logger.info("CIRCULARITY (1.0 = perfect circle):")
    logger.info(f"  Min: {stats['circularity']['min']:>6.4f}  |  Max: {stats['circularity']['max']:>6.4f}  |  Mean: {stats['circularity']['mean']:>6.4f}")
    if stats['circularity']['mean'] < 0.8:
        logger.info(f"  → Mostly non-circular shapes")
    elif stats['circularity']['mean'] > 0.9:
        logger.info(f"  → Mostly circular shapes")
    else:
        logger.info(f"  → Moderately circular shapes")
    logger.info("")

    rec = stats["tile_recommendation"]
    logger.info("=" * 80)
    logger.info("TILE SIZE RECOMMENDATION")
    logger.info("=" * 80)
    logger.info(f"Recommended tile size: {rec['recommended_size']} × {rec['recommended_size']} pixels")
    logger.info(f"Coverage: {rec['coverage_percentage']:.0f}% of glomeruli ({rec['glomeruli_covered']}/{stats['count']})")
    logger.info("=" * 80)




# ===========
# EJECUCIÓN
# ===========


if __name__ == "__main__":
    INPUT_DIR = "/Users/olivera/Documents/Proyecto_Final_Glomerulos/Entradas"
    OUTPUT_BASE = "/Users/olivera/Documents/Proyecto_Final_Glomerulos/Salidas/tamanio"
    ZOOM_SCALE = 0.5  # Must match zoom_scale in tiles.py

    logger.info("=" * 70)
    logger.info("TAMANIO: Análisis de distribución de tamaños de glomérulos")
    logger.info("=" * 70)
    logger.info(f"zoom_scale = {ZOOM_SCALE}  (tile px = native px × {ZOOM_SCALE})")

    # Buscar archivos GeoJSON
    logger.info(f"\n[1] Buscando archivos GeoJSON en {INPUT_DIR}...")
    geojson_files = find_geojson_files(INPUT_DIR)

    if not geojson_files:
        logger.error("No se encontraron archivos GeoJSON.")
        exit(1)

    logger.info(f"Encontrados {len(geojson_files)} archivos GeoJSON:")
    for geojson_path in geojson_files:
        logger.info(f"    - {Path(geojson_path).name}")

    # Crear directorio de salida
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Acumular todos los glomérulos sin análisis por-biopsia
    logger.info("\n[2] Leyendo y consolidando glomérulos (parallelizado)...")

    # Ordenar archivos por tamaño (más grandes primero para mejor load balancing)
    geojson_files_sorted = sorted(
        geojson_files,
        key=lambda x: os.path.getsize(x),
        reverse=True
    )

    # Preparar argumentos para workers
    tasks = [(geojson_path, ZOOM_SCALE) for geojson_path in geojson_files_sorted]

    # Calcular número seguro de workers
    num_workers = _safe_worker_count(geojson_files_sorted)
    logger.info(f"Iniciando procesamiento de {len(geojson_files)} archivos GeoJSON")
    logger.info(f"Usando {num_workers} procesos en paralelo...\n")

    start_time = time.time()
    all_rows = []

    with Pool(num_workers) as pool:
        for idx, (geojson_path, rows) in enumerate(pool.imap_unordered(_process_single_geojson, tasks), 1):
            dataset_name = Path(geojson_path).stem
            dataset_count = len(rows)
            all_rows.extend(rows)

            if dataset_count > 0:
                logger.info(f"  [{idx}/{len(geojson_files)}] {dataset_name}: {dataset_count} glomérulos")
            else:
                logger.info(f"  [{idx}/{len(geojson_files)}] {dataset_name}: SKIP - sin geometrías")

    if not all_rows:
        logger.error("No se pudieron procesar glomérulos.")
        exit(1)

    # Crear DataFrame consolidado
    df_consolidated = pd.DataFrame(all_rows)
    logger.info(f"\nTotal: {len(df_consolidated)} glomérulos de {len(geojson_files)} biopsias")

    # Guardar CSV consolidado
    consolidated_csv = os.path.join(OUTPUT_BASE, "glomeruli_sizes_consolidated.csv")
    df_consolidated.to_csv(consolidated_csv, index=False)
    logger.info(f"     CSV guardado: {consolidated_csv}")

    # Análisis consolidado (una sola vez)
    logger.info("\n[3] Generando análisis consolidado...")

    stats = {
        "zoom_scale": ZOOM_SCALE,
        "count": len(df_consolidated),
        "square_size_native": {
            "min": float(df_consolidated["square_size_native_px"].min()),
            "max": float(df_consolidated["square_size_native_px"].max()),
            "mean": float(df_consolidated["square_size_native_px"].mean()),
            "std": float(df_consolidated["square_size_native_px"].std())
        },
        "square_size_tile": {
            "min": float(df_consolidated["square_size_tile_px"].min()),
            "max": float(df_consolidated["square_size_tile_px"].max()),
            "mean": float(df_consolidated["square_size_tile_px"].mean()),
            "std": float(df_consolidated["square_size_tile_px"].std())
        },
        "area": {
            "min": float(df_consolidated["area_tile_px2"].min()),
            "max": float(df_consolidated["area_tile_px2"].max()),
            "mean": float(df_consolidated["area_tile_px2"].mean()),
            "std": float(df_consolidated["area_tile_px2"].std())
        },
        "width": {
            "min": float(df_consolidated["width_tile_px"].min()),
            "max": float(df_consolidated["width_tile_px"].max()),
            "mean": float(df_consolidated["width_tile_px"].mean()),
            "std": float(df_consolidated["width_tile_px"].std())
        },
        "height": {
            "min": float(df_consolidated["height_tile_px"].min()),
            "max": float(df_consolidated["height_tile_px"].max()),
            "mean": float(df_consolidated["height_tile_px"].mean()),
            "std": float(df_consolidated["height_tile_px"].std())
        },
        "circularity": {
            "min": float(df_consolidated["circularity"].min()),
            "max": float(df_consolidated["circularity"].max()),
            "mean": float(df_consolidated["circularity"].mean())
        }
    }

    percentiles = calculate_tile_percentiles(df_consolidated["square_size_tile_px"].tolist())
    stats["percentiles"] = percentiles

    p95_tile = percentiles["p95"]
    recommended_size = recommend_tile_size(p95_tile)
    coverage_pct = calculate_coverage_percentage(df_consolidated["square_size_tile_px"].tolist(), recommended_size)

    stats["tile_recommendation"] = {
        "p95_raw": p95_tile,
        "recommended_size": recommended_size,
        "coverage_percentage": coverage_pct,
        "glomeruli_covered": int(np.ceil(len(df_consolidated) * coverage_pct / 100.0)),
        "glomeruli_incomplete": len(df_consolidated) - int(np.ceil(len(df_consolidated) * coverage_pct / 100.0))
    }

    # Generar visualizaciones
    _create_glomeruli_plots(df_consolidated, stats, OUTPUT_BASE)

    # Generar reportes
    _generate_tile_report("CONSOLIDADO", df_consolidated, stats, OUTPUT_BASE)
    _print_glomeruli_summary("CONSOLIDADO", stats)

    logger.info("\n" + "=" * 70)
    logger.info("DONE - Análisis completado")
    logger.info("=" * 70)