#!/usr/bin/env python3
"""
Estandarización Z-Score de tiles normalizados.

Tercer paso del pipeline:
1. tiles.py → tiling con filtrado de tejido
2. normalizacion.py → normalización de Reinhard por colores
3. estandarizacion.py → Z-Score por canal → PNG uint8 con reescalado reversible

Entrada: PNG uint8 normalizados (Salidas/Normalizados/)
Salida: PNG uint8 estandarizados (Salidas/Estandarizados/)

El Z-Score se calcula solo sobre píxeles de tejido, para evitar que el fondo blanco
sesgue la media/std. El template de estadísticas se cachea en .estandarizacion/

Reescalado reversible:
  Z-Score float32 → uint8: uint8 = (float32 + 4) / 8 * 255
  uint8 → float32 back: float32 = (uint8 / 255 * 8) - 4

Esto preserva Z-Score ∈ [-4, +4] en PNG uint8 comprimido.
"""

import argparse
import json
import logging
import multiprocessing
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_tissue_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Máscara de tejido: L<230 (LAB) + S>10 (HSV) + morfología.

    Reutiliza la lógica de normalizacion.py para consistencia.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    l_channel = lab[..., 0]
    s_channel = hsv[..., 1]

    mask = (l_channel < 230) & (s_channel > 10)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask.astype(bool)


def leer_imagen(path: Path) -> np.ndarray:
    """Lee imagen desde ruta con soporte a caracteres especiales (Windows)."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def compute_channel_stats(input_dir: Path, n_samples: int = 200) -> Tuple[list, list]:
    """
    Calcula media y desviación estándar por canal (RGB) sobre una muestra de tiles.

    Usa Welford's online algorithm para evitar acumular todos los píxeles en memoria.
    Solo usa píxeles de tejido para evitar sesgo del fondo blanco.
    Retorna (means, stds) como listas de 3 floats cada una.
    """
    logger.info(f"Buscando tiles en {input_dir} para calcular estadísticas...")

    all_tiles = list(input_dir.rglob("*.png"))
    logger.info(f"Total de tiles encontrados: {len(all_tiles)}")

    if len(all_tiles) < 10:
        raise ValueError(
            f"Se necesitan al menos 10 tiles para calcular estadísticas. "
            f"Encontrados: {len(all_tiles)}"
        )

    sample_tiles = random.sample(all_tiles, min(n_samples, len(all_tiles)))
    logger.info(f"Muestreando {len(sample_tiles)} tiles para estadísticas...")

    tile_means_list = [[], [], []]
    tile_vars_list = [[], [], []]
    tile_counts = []

    for i, tile_path in enumerate(sample_tiles):
        if (i + 1) % 50 == 0:
            logger.info(f"  Procesados {i + 1}/{len(sample_tiles)}...")

        img_bgr = leer_imagen(tile_path)
        if img_bgr is None:
            logger.warning(f"  No se pudo leer {tile_path}, saltando...")
            continue

        mask = get_tissue_mask(img_bgr)
        if mask.sum() < 100:
            logger.debug(f"  {tile_path.name}: <100 píxeles de tejido, saltando...")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0

        n_pixels = mask.sum()
        tile_counts.append(n_pixels)

        for c in range(3):
            tissue_pixels = img_float[..., c][mask]
            tile_means_list[c].append(np.mean(tissue_pixels))
            tile_vars_list[c].append(np.var(tissue_pixels))

    if not tile_counts:
        raise ValueError(
            "No se encontraron suficientes píxeles de tejido en la muestra. "
            "Verifica que los tiles tengan contenido válido."
        )

    means = []
    stds = []
    for c in range(3):
        tile_means_arr = np.array(tile_means_list[c])
        tile_vars_arr = np.array(tile_vars_list[c])
        counts_arr = np.array(tile_counts)

        overall_mean = np.average(tile_means_arr, weights=counts_arr)
        overall_var = np.average(
            tile_vars_arr + (tile_means_arr - overall_mean) ** 2, weights=counts_arr
        )
        overall_std = np.sqrt(overall_var)

        means.append(float(overall_mean))
        stds.append(float(overall_std))

    logger.info("Estadísticas calculadas (RGB) - Welford online:")
    logger.info(f"  Medias (R,G,B): {[f'{m:.4f}' for m in means]}")
    logger.info(f"  Stds   (R,G,B): {[f'{s:.4f}' for s in stds]}")

    return means, stds


def load_or_compute_stats(
    input_dir: Path, n_samples: int = 200, rebuild: bool = False
) -> Tuple[list, list]:
    """
    Carga estadísticas desde caché o las calcula si no existen.

    Retorna (means, stds) como listas de 3 floats.
    """
    cache_dir = input_dir.parent / ".estandarizacion"
    cache_file = cache_dir / "channel_stats.json"

    if cache_file.exists() and not rebuild:
        logger.info(f"Cargando estadísticas desde {cache_file}...")
        with open(cache_file, "r") as f:
            data = json.load(f)
        return data["mean"], data["std"]

    logger.info("Calculando nuevas estadísticas...")
    means, stds = compute_channel_stats(input_dir, n_samples)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({"mean": means, "std": stds}, f, indent=2)

    logger.info(f"Estadísticas guardadas en {cache_file}")
    return means, stds


def apply_zscore(
    img_bgr: np.ndarray, means: list, stds: list
) -> np.ndarray:
    """
    Aplica Z-Score por canal y convierte BGR→RGB.

    1. Convierte a RGB (convención PyTorch/torchvision)
    2. Escala a [0, 1]
    3. Aplica Z-Score sobre TODOS los píxeles: (x - mean) / std
       Stats calculadas sobre tejido, aplicadas a toda la imagen.
       El fondo queda como el Z-Score de su valor original (blanco → valor positivo alto),
       lo que preserva su identidad y permite al modelo distinguir tejido de fondo.

    Retorna float32 (H, W, 3) en orden RGB.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_float = img_rgb.astype(np.float32) / 255.0

    result = np.empty_like(img_float, dtype=np.float32)

    for c in range(3):
        result[..., c] = (img_float[..., c] - means[c]) / (stds[c] + 1e-5)

    return result


def guardar_imagen_zscore_png(path: Path, img_float32: np.ndarray) -> bool:
    """
    Guarda Z-Score float32 como PNG uint8 con reescalado reversible.

    Z-Score típico: [-3, +3]. Reescalamos a [0, 255]:
      uint8 = (float32 + 4) / 8 * 255 ≈ uint8 ∈ [0, 255]
      float32_back = (uint8 / 255 * 8) - 4

    Esto preserva la distribución con pérdida mínima de precisión.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        img_uint8 = np.clip((img_float32 + 4.0) / 8.0 * 255.0, 0, 255).astype(
            np.uint8
        )

        success = cv2.imwrite(str(path), img_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        return bool(success)
    except Exception:
        return False


def process_single_tile(args: Tuple) -> Tuple[str, bool]:
    """Worker: procesa un tile individual."""
    tile_path, output_path, means, stds = args

    try:
        img_bgr = leer_imagen(tile_path)
        if img_bgr is None:
            return str(tile_path), False

        img_estandarizado = apply_zscore(img_bgr, means, stds)

        if not guardar_imagen_zscore_png(output_path, img_estandarizado):
            return str(tile_path), False

        return str(tile_path), True

    except Exception as e:
        logger.error(f"Error procesando {tile_path}: {e}")
        return str(tile_path), False


def calculate_optimal_workers(avg_tile_size_mb: float = 3.0) -> int:
    """Calcula nº de workers basado en RAM disponible (70% máximo)."""
    mem = psutil.virtual_memory()
    free_mb = mem.available / (1024**2)
    available_for_workers = free_mb * 0.7

    workers = max(1, int(available_for_workers / (avg_tile_size_mb * 5)))
    workers = min(workers, multiprocessing.cpu_count())

    logger.info(f"RAM disponible: {free_mb:.0f}MB → {workers} workers")
    return workers


def process_folder(
    input_dir: Path,
    output_dir: Path,
    workers: int = None,
    n_samples: int = 200,
    rebuild: bool = False,
):
    """Orquesta el procesamiento de todos los tiles."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise ValueError(f"Directorio de entrada no existe: {input_dir}")

    tile_paths = sorted(input_dir.rglob("*.png"))
    logger.info(f"Total de tiles a procesar: {len(tile_paths)}")

    if len(tile_paths) == 0:
        logger.warning("No se encontraron tiles PNG. Nada que hacer.")
        return

    means, stds = load_or_compute_stats(input_dir, n_samples, rebuild)

    if workers is None:
        workers = calculate_optimal_workers()

    tile_args = []
    for tile_path in tile_paths:
        rel_path = tile_path.relative_to(input_dir)
        output_path = (output_dir / rel_path).with_suffix(".png")
        tile_args.append((tile_path, output_path, means, stds))

    processed = 0
    failed = 0

    logger.info(f"Iniciando procesamiento con {workers} workers...")

    with multiprocessing.Pool(workers) as pool:
        for tile_path_str, success in pool.imap_unordered(
            process_single_tile, tile_args, chunksize=10
        ):
            if success:
                processed += 1
            else:
                failed += 1

            if (processed + failed) % 50 == 0:
                logger.info(
                    f"Progreso: {processed + failed}/{len(tile_paths)} "
                    f"({processed} OK, {failed} fallos)"
                )

    logger.info(
        f"Procesamiento completado: {processed}/{len(tile_paths)} tiles "
        f"({failed} fallos)"
    )
    logger.info(f"Tiles estandarizados guardados en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Estandarización Z-Score de tiles normalizados."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Salidas/Normalizados",
        help="Directorio de tiles normalizados (entrada)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Salidas/Estandarizados",
        help="Directorio de tiles estandarizados (salida)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de procesos paralelos (auto si no se especifica)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Número de tiles a muestrear para calcular estadísticas",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recalcular estadísticas aunque el caché exista",
    )

    args = parser.parse_args()

    process_folder(
        input_dir=args.input,
        output_dir=args.output,
        workers=args.workers,
        n_samples=args.n_samples,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
