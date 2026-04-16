import os
import math
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape


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


def find_tiff_geojson_pairs(input_dir: str = r"D:\Anotaciones\Entradas") -> List[Tuple[str, str]]:
    """
    Busca automáticamente en input_dir todos los pares TIFF-GeoJSON.
    Retorna lista de tuplas (tiff_path, geojson_path).

    Ejemplos de patrones:
    - BR-007-HYE-25-CONV.tiff + BR-007-PAS-25-CONV.geojson
    - imagen.tiff + imagen.geojson
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"[WARN] Directorio {input_dir} no existe.")
        return []

    tiff_files = set(input_path.glob("*.tiff")) | set(input_path.glob("*.TIF"))
    geojson_files = set(input_path.glob("*.geojson")) | set(input_path.glob("*.json"))

    pairs = []

    for tiff in sorted(tiff_files):
        tiff_stem = tiff.stem

        # Intenta coincidencia exacta (mismo nombre)
        for geojson in geojson_files:
            if geojson.stem == tiff_stem:
                pairs.append((str(tiff), str(geojson)))
                break
        else:
            # Intenta coincidencia parcial (comparte prefijo común, ej: BR-007)
            for geojson in geojson_files:
                # Extrae prefijo: BR-007 de BR-007-HYE-25-CONV
                tiff_parts = tiff_stem.split("-")
                geo_parts = geojson.stem.split("-")
                if len(tiff_parts) >= 2 and len(geo_parts) >= 2:
                    if tiff_parts[0] == geo_parts[0] and tiff_parts[1] == geo_parts[1]:
                        pairs.append((str(tiff), str(geojson)))
                        break

    return pairs




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
    candidates = [128, 256, 384, 512, 768, 1024, 1536, 2048]

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
                           tiff_path: Optional[str] = None,
                           output_dir: str = "analysis_plots",
                           output_csv: str = "glomeruli_sizes.csv") -> Tuple[pd.DataFrame, Dict]:
    """
    Mide el tamaño de cada glomérulo en un GeoJSON.
    Calcula tamaño de tile cuadrado recomendado basado en percentil 95%.

    Retorna:
    - DataFrame: una fila por glomérulo (nombre, área, ancho, alto, perímetro, circularidad, square_size)
    - Dict: estadísticas agregadas (count, min, max, mean, std por métrica, + percentiles y recomendación)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Cargar GeoJSON
    geojson_obj = load_geojson(geojson_path)
    features = geojson_obj.get("features", [])

    if not features:
        print(f"[WARN] No hay geometrías en {geojson_path}")
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

            rows.append({
                "name": glomeruli_name,
                "area_pixels": round(area, 2),
                "width_pixels": round(width, 2),
                "height_pixels": round(height, 2),
                "square_size": round(square_size, 2),
                "perimeter_pixels": round(perimeter, 2),
                "circularity": round(circularity, 4),
                "feature_index": idx
            })

        except Exception as e:
            print(f"[WARN] Error procesando feature {idx}: {e}")

    if not rows:
        print("[ERROR] No se pudieron procesar geometrías válidas.")
        return pd.DataFrame(), {}

    # Crear DataFrame
    df = pd.DataFrame(rows)

    # Calcular estadísticas base
    stats = {
        "count": len(df),
        "area": {
            "min": float(df["area_pixels"].min()),
            "max": float(df["area_pixels"].max()),
            "mean": float(df["area_pixels"].mean()),
            "std": float(df["area_pixels"].std())
        },
        "width": {
            "min": float(df["width_pixels"].min()),
            "max": float(df["width_pixels"].max()),
            "mean": float(df["width_pixels"].mean()),
            "std": float(df["width_pixels"].std())
        },
        "height": {
            "min": float(df["height_pixels"].min()),
            "max": float(df["height_pixels"].max()),
            "mean": float(df["height_pixels"].mean()),
            "std": float(df["height_pixels"].std())
        },
        "square_size": {
            "min": float(df["square_size"].min()),
            "max": float(df["square_size"].max()),
            "mean": float(df["square_size"].mean()),
            "std": float(df["square_size"].std())
        },
        "perimeter": {
            "min": float(df["perimeter_pixels"].min()),
            "max": float(df["perimeter_pixels"].max()),
            "mean": float(df["perimeter_pixels"].mean()),
            "std": float(df["perimeter_pixels"].std())
        },
        "circularity": {
            "min": float(df["circularity"].min()),
            "max": float(df["circularity"].max()),
            "mean": float(df["circularity"].mean())
        }
    }

    # Calcular percentiles de tamaño cuadrado
    percentiles = calculate_tile_percentiles(df["square_size"].tolist())
    stats["percentiles"] = percentiles

    # Calcular tamaño de tile recomendado
    p95_size = percentiles["p95"]
    recommended_size = recommend_tile_size(p95_size)
    coverage_pct = calculate_coverage_percentage(df["square_size"].tolist(), recommended_size)

    stats["tile_recommendation"] = {
        "p95_raw": p95_size,
        "recommended_size": recommended_size,
        "coverage_percentage": coverage_pct,
        "glomeruli_covered": int(np.ceil(len(df) * coverage_pct / 100.0)),
        "glomeruli_incomplete": len(df) - int(np.ceil(len(df) * coverage_pct / 100.0))
    }

    # Guardar CSV
    df.to_csv(output_csv, index=False)
    print(f"[OK] DataFrame de glomérulos guardado en {output_csv}")

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

    # 1. Histograma de áreas
    plt.figure(figsize=(10, 6))
    plt.hist(df["area_pixels"], bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Area (pixels²)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title("Distribution of Glomeruli Areas", fontsize=13, fontweight="bold")
    plt.axvline(stats["area"]["mean"], color="red", linestyle="--", linewidth=2, label=f"Mean: {stats['area']['mean']:.0f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    area_plot = os.path.join(output_dir, "01_histogram_areas.png")
    plt.savefig(area_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot guardado: {area_plot}")

    # 2. Scatter: width vs height
    plt.figure(figsize=(10, 6))
    plt.scatter(df["width_pixels"], df["height_pixels"], s=100, alpha=0.6, color="darkgreen", edgecolors="black")
    plt.xlabel("Width (pixels)", fontsize=11)
    plt.ylabel("Height (pixels)", fontsize=11)
    plt.title("Glomeruli Dimensions: Width vs Height", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    # Añade línea de tendencia si hay variabilidad
    z = np.polyfit(df["width_pixels"], df["height_pixels"], 1)
    p = np.poly1d(z)
    plt.plot(df["width_pixels"], p(df["width_pixels"]), "r--", linewidth=2, label="Trend")
    plt.legend()
    scatter_plot = os.path.join(output_dir, "02_scatter_dimensions.png")
    plt.savefig(scatter_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot guardado: {scatter_plot}")

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
    print(f"[OK] Plot guardado: {circ_plot}")

    # 4. Histograma de tamaño cuadrado con percentiles
    plt.figure(figsize=(12, 7))
    square_sizes = df["square_size"].tolist()
    bins = max(15, len(set(square_sizes)) // 2)
    plt.hist(square_sizes, bins=bins, color="skyblue", edgecolor="black", alpha=0.7, label="Distribution")

    # Línea de media (azul)
    mean_size = stats["square_size"]["mean"]
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

    plt.xlabel("Square Tile Size (pixels)", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency (number of glomeruli)", fontsize=12, fontweight="bold")
    plt.title("Distribution of Square Tile Sizes for Glomeruli Coverage", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, alpha=0.3)

    square_plot = os.path.join(output_dir, "04_histogram_square_sizes.png")
    plt.savefig(square_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Plot guardado: {square_plot}")


def _generate_tile_report(dataset_name: str, df: pd.DataFrame, stats: Dict, output_dir: str):
    """Genera un archivo de reporte de texto con recomendación de tamaño de tile."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("GLOMERULI SIZE DISTRIBUTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Dataset: {dataset_name}")
    report_lines.append(f"Total glomeruli analyzed: {stats['count']}")
    report_lines.append("")

    # Advertencia si dataset es pequeño
    if stats['count'] < 10:
        report_lines.append("[WARNING] Dataset tiene menos de 10 glomérulos. Las estadísticas pueden no ser representativas.")
        report_lines.append("")

    report_lines.append("SQUARE TILE SIZE (pixels × pixels):")
    report_lines.append("")
    report_lines.append("Statistics:")
    report_lines.append(f"  Min:        {stats['square_size']['min']:.0f} px")
    report_lines.append(f"  Max:        {stats['square_size']['max']:.0f} px")
    report_lines.append(f"  Mean:       {stats['square_size']['mean']:.0f} px (±{stats['square_size']['std']:.0f})")
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
    conservative_size = recommend_tile_size(stats['square_size']['max'])
    if conservative_size > rec['recommended_size']:
        conservative_cov = calculate_coverage_percentage(df["square_size"].tolist(), conservative_size)
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

    print(f"[OK] Reporte guardado: {report_path}")
    return report_text


def _print_glomeruli_summary(dataset_name: str, stats: Dict):
    """Imprime un resumen formateado de estadísticas, incluyendo recomendación de tile."""
    print("\n" + "=" * 80)
    print("GLOMERULI SIZE ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Total glomeruli: {stats['count']}")
    print()

    # Advertencia si dataset es pequeño
    if stats['count'] < 10:
        print("[WARNING] Dataset tiene menos de 10 glomérulos. Las estadísticas pueden no ser representativas.")
        print()

    print("AREA (pixels²):")
    print(f"  Min: {stats['area']['min']:>12,.0f}  |  Max: {stats['area']['max']:>12,.0f}")
    print(f"  Mean: {stats['area']['mean']:>10,.0f} ± {stats['area']['std']:.0f}")
    print()

    print("SQUARE TILE SIZE (pixels):")
    print(f"  Min: {stats['square_size']['min']:>10,.0f}  |  Max: {stats['square_size']['max']:>10,.0f}")
    print(f"  Mean: {stats['square_size']['mean']:>10,.0f} ± {stats['square_size']['std']:.0f}")
    print()
    print("  Percentiles:")
    print(f"    P50: {stats['percentiles']['p50']:>8,.0f} px  |  P75: {stats['percentiles']['p75']:>8,.0f} px  |  P85: {stats['percentiles']['p85']:>8,.0f} px")
    print(f"    P90: {stats['percentiles']['p90']:>8,.0f} px  |  P95: {stats['percentiles']['p95']:>8,.0f} px")
    print()

    print("DIMENSIONS (pixels):")
    print(f"  Width:")
    print(f"    Min: {stats['width']['min']:>10,.0f}  |  Max: {stats['width']['max']:>10,.0f}  |  Mean: {stats['width']['mean']:>10,.0f} ± {stats['width']['std']:.0f}")
    print(f"  Height:")
    print(f"    Min: {stats['height']['min']:>10,.0f}  |  Max: {stats['height']['max']:>10,.0f}  |  Mean: {stats['height']['mean']:>10,.0f} ± {stats['height']['std']:.0f}")
    print()

    print("CIRCULARITY (1.0 = perfect circle):")
    print(f"  Min: {stats['circularity']['min']:>6.4f}  |  Max: {stats['circularity']['max']:>6.4f}  |  Mean: {stats['circularity']['mean']:>6.4f}")
    if stats['circularity']['mean'] < 0.8:
        print(f"  → Mostly non-circular shapes")
    elif stats['circularity']['mean'] > 0.9:
        print(f"  → Mostly circular shapes")
    else:
        print(f"  → Moderately circular shapes")
    print()

    rec = stats["tile_recommendation"]
    print("=" * 80)
    print("TILE SIZE RECOMMENDATION")
    print("=" * 80)
    print(f"Recommended tile size: {rec['recommended_size']} × {rec['recommended_size']} pixels")
    print(f"Coverage: {rec['coverage_percentage']:.0f}% of glomeruli ({rec['glomeruli_covered']}/{stats['count']})")
    print("=" * 80)




# ===========
# EJECUCIÓN
# ===========


if __name__ == "__main__":
    print("=" * 70)
    print("TAMANIO: Análisis de distribución de tamaños de glomérulos")
    print("=" * 70)

    # Buscar automáticamente pares TIFF-GeoJSON
    print("\n[1] Buscando pares TIFF-GeoJSON en D:\\Anotaciones\\Entradas...")
    pairs = find_tiff_geojson_pairs()

    if not pairs:
        print("[ERROR] No se encontraron pares TIFF-GeoJSON.")
        exit(1)

    print(f"[OK] Encontrados {len(pairs)} pares:")
    for tiff, geojson in pairs:
        print(f"    - {Path(tiff).name} <-> {Path(geojson).name}")

    # Analizar tamaños de glomérulos
    print("\n[2] Analizando tamaños de glomérulos...")
    all_glomeruli_data = []

    for tiff_path, geojson_path in pairs:
        dataset_name = Path(geojson_path).stem
        print(f"\n[*] Procesando: {dataset_name}")

        try:
            df_glom, stats_glom = analyze_glomeruli_sizes(
                geojson_path=geojson_path,
                tiff_path=tiff_path,
                output_dir="analysis_plots",
                output_csv=f"glomeruli_sizes_{dataset_name}.csv"
            )
            if not df_glom.empty:
                all_glomeruli_data.append(df_glom)
        except Exception as e:
            print(f"[ERROR] {e}")

    # Guardar tabla consolidada
    if all_glomeruli_data:
        df_consolidated = pd.concat(all_glomeruli_data, ignore_index=True)
        df_consolidated.to_csv("glomeruli_sizes_consolidated.csv", index=False)
        print(f"\n[OK] Tabla consolidada: {len(df_consolidated)} glomérulos")
        print(f"     guardada en: glomeruli_sizes_consolidated.csv")

    print("\n" + "=" * 70)
    print("[DONE] Análisis completado")
    print("=" * 70)