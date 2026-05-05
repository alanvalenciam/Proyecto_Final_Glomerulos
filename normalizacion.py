import os
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _safe_worker_count(tile_paths):
    """
    Compute safe worker count based on available RAM and tile sizes.

    PNG tiles are small (~0.5-2 MB), but in memory they're ~3 channels × 3 bytes per pixel.
    Expansion factor: 2x (PNG small, but decompressed + processing buffers).
    Uses psutil if available, falls back to conservative estimate.
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        available_gb = 2.0  # Conservative fallback

    max_file_size_gb = 0.0
    for path in tile_paths:
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            max_file_size_gb = max(max_file_size_gb, size_gb)

    if max_file_size_gb == 0:
        max_file_size_gb = 0.005  # ~5MB fallback for small tiles

    expansion_factor = 2  # PNGs decompress to ~2x in memory
    memory_per_worker = max_file_size_gb * expansion_factor

    if memory_per_worker > 0:
        num_workers = int((available_gb * 0.7) / memory_per_worker)
        num_workers = max(1, min(num_workers, cpu_count()))
    else:
        num_workers = min(4, cpu_count())

    logger.info(f"RAM-aware: {available_gb:.2f}GB available, max file {max_file_size_gb:.6f}GB → {num_workers} workers")
    return num_workers


def leer_imagen(ruta):
    """Lee imágenes en rutas con tildes o caracteres especiales en Windows"""
    return cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)

def guardar_imagen(ruta, imagen):
    """Guarda imágenes en rutas con tildes o caracteres especiales en Windows"""
    cv2.imencode('.png', imagen)[1].tofile(ruta)

def get_lab_stats(image_bgr):
    """Calcula la media y desviación estándar en el espacio de color LAB"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    l_mean, l_std = l.mean(), l.std()
    a_mean, a_std = a.mean(), a.std()
    b_mean, b_std = b.mean(), b.std()
    
    return (l_mean, a_mean, b_mean), (l_std, a_std, b_std)

def apply_reinhard(source_bgr, target_means, target_stds):
    """Aplica la fórmula matemática de Reinhard a una imagen"""
    source_means, source_stds = get_lab_stats(source_bgr)

    lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    l = ((l - source_means[0]) * (target_stds[0] / (source_stds[0] + 1e-5))) + target_means[0]
    a = ((a - source_means[1]) * (target_stds[1] / (source_stds[1] + 1e-5))) + target_means[1]
    b = ((b - source_means[2]) * (target_stds[2] / (source_stds[2] + 1e-5))) + target_means[2]

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    lab_normalized = cv2.merge((l, a, b)).astype(np.uint8)
    return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)


def _normalize_single_tile(args):
    """
    Normalize a single tile using Reinhard stain normalization.

    Args:
        args: Tuple of (source_path, output_path, target_means, target_stds)

    Returns:
        str: source_path if successful, None if failed.
    """
    source_path, output_path, target_means, target_stds = args

    try:
        img = leer_imagen(source_path)
        if img is None:
            logger.warning(f"Failed to read: {source_path}")
            return None

        norm_img = apply_reinhard(img, target_means, target_stds)
        guardar_imagen(output_path, norm_img)
        return str(source_path)

    except Exception as e:
        logger.error(f"Error normalizing {source_path}: {e}", exc_info=True)
        return None


def process_folder(input_dir, output_dir, template_path, num_workers=None):
    """
    Normalize tiles using Reinhard stain normalization with parallel processing.

    Args:
        input_dir: Directory containing input tiles
        output_dir: Directory for output normalized tiles
        template_path: Path to reference template image
        num_workers: Number of parallel workers (None = auto-calculate based on RAM)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading reference template...")
    template_img = leer_imagen(template_path)
    if template_img is None:
        logger.error(f"Failed to load template: {template_path}")
        return

    target_means, target_stds = get_lab_stats(template_img)
    logger.info(f"Template analyzed successfully.\n")

    # Only accept PNG/JPG (small tiles)
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    archivos = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

    if not archivos:
        logger.warning(f"No tiles (.png or .jpg) found in: {input_dir}")
        return

    logger.info(f"Found {len(archivos)} tile(s) to normalize")

    # Sort by file size (descending) for load balancing
    tile_paths = [os.path.join(input_dir, f) for f in archivos]
    tile_paths.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0, reverse=True)
    archivos = [os.path.basename(p) for p in tile_paths]

    # Calculate num_workers if not provided
    if num_workers is None:
        num_workers = _safe_worker_count(tile_paths)

    # Build task list for multiprocessing
    tasks = [
        (os.path.join(input_dir, file_name),
         os.path.join(output_dir, file_name),
         target_means,
         target_stds)
        for file_name in archivos
    ]

    logger.info(f"Starting parallel normalization with {num_workers} workers...")
    start_time = time.time()

    # Process tiles in parallel
    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_normalize_single_tile, tasks, chunksize=10), 1):
            if result:
                logger.info(f"Tile {i}/{len(tasks)} completed")
            else:
                logger.warning(f"Tile {i}/{len(tasks)} failed")

    elapsed = time.time() - start_time
    logger.info(f"Complete! {len(archivos)} tiles normalized in {elapsed:.2f}s")

if __name__ == "__main__":
    import argparse

    # Defaults
    base_dir = Path("/Users/olivera/Documents/Proyecto_Final_Glomerulos/Salidas/Imagen")
    default_input = str(base_dir)
    default_output = str(base_dir.parent / "Normalizados")
    default_template = "/Users/olivera/Documents/Proyecto_Final_Glomerulos/referencia_reinhard.png"

    parser = argparse.ArgumentParser(
        description="Normalización de Reinhard para tiles histopatológicos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uso por defecto (sin argumentos):
  python3 normalizacion.py

Rutas por defecto:
  --input    """ + default_input + """
  --output   """ + default_output + """
  --template """ + default_template + """

Uso personalizado:
  python3 normalizacion.py \\
    --input /ruta/a/tiles \\
    --output /ruta/salida \\
    --template /ruta/referencia.png
        """
    )

    parser.add_argument(
        "--input",
        default=default_input,
        help=f"Carpeta raíz con tiles (default: {default_input})"
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Carpeta de salida para tiles normalizados (default: {default_output})"
    )
    parser.add_argument(
        "--template",
        default=default_template,
        help=f"Ruta al tile de referencia (default: {default_template})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de workers paralelos (default: auto-calcula basado en RAM)"
    )

    args = parser.parse_args()
    process_folder(args.input, args.output, args.template, num_workers=args.workers)