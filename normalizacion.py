import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import cpu_count
from collections import deque
import logging
import time
import random
import json
import shutil
from typing import Optional, Tuple, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _safe_worker_count(tile_size_hint: int = 1024) -> int:
    """
    Compute safe worker count based on available RAM and tile pixel dimensions.

    Memory per worker = tile_size² × 3 channels × 6 bytes
    (uint8 input + float32 LAB 4x + output + mask ≈ 18–30 MB per 1024×1024 tile).
    Uses psutil if available, falls back to conservative estimate.
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        available_gb = 2.0  # Conservative fallback

    # Pixel-based RAM estimation: uint8 + float32 LAB + output + mask
    bytes_per_worker = tile_size_hint * tile_size_hint * 3 * 6
    gb_per_worker = bytes_per_worker / (1024**3)

    if gb_per_worker > 0:
        num_workers = int((available_gb * 0.80) / gb_per_worker)
        num_workers = max(1, min(num_workers, cpu_count()))
    else:
        num_workers = min(4, cpu_count())

    logger.info(f"RAM-aware: {available_gb:.2f}GB available, ~{gb_per_worker*1024:.1f}MB per worker → {num_workers} workers")
    return num_workers


def leer_imagen(ruta: str) -> Optional[np.ndarray]:
    """Lee imágenes en rutas con tildes o caracteres especiales en Windows"""
    return cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)

def guardar_imagen(ruta: str, imagen: np.ndarray) -> None:
    """Guarda imágenes en rutas con tildes o caracteres especiales en Windows"""
    cv2.imencode('.png', imagen)[1].tofile(ruta)

def get_tissue_mask(image_bgr: np.ndarray, bg_l_threshold: int = 230, min_saturation: int = 10) -> np.ndarray:
    """
    Generate binary tissue mask combining LAB L and HSV S channels + morphology.

    L channel alone (L < threshold) detects white background but misses dark noise (ink, dust, folds).
    Combined with saturation (S > threshold) to exclude desaturated dark artifacts.
    Apply morphological operations to clean up the mask.

    Returns boolean mask where True = tissue pixel, False = background.
    """
    # L channel: exclude bright white background
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_mask = lab[:, :, 0] < bg_l_threshold

    # Saturation channel: exclude low-saturation dark noise (ink, dust have low S)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s_mask = hsv[:, :, 1] > min_saturation

    # Combine: pixel must be dark AND have color → tissue
    combined = (l_mask & s_mask).astype(np.uint8) * 255

    # Morphological closing: fill small holes in tissue (nuclei, vessels)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)

    # Morphological opening: remove small isolated artifacts (dust, noise)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    return cleaned.astype(bool)

def get_lab_stats(image_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute LAB mean and std. If mask provided, use only tissue pixels.

    Args:
        image_bgr: Input BGR image
        mask: Optional boolean mask (True = tissue, False = background).
              If provided, stats are computed only on masked pixels.

    Returns:
        ((l_mean, a_mean, b_mean), (l_std, a_std, b_std))

    Raises:
        ValueError: If tissue pixels < 100 (insufficient data).
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    if mask is not None:
        l = l[mask]
        a = a[mask]
        b = b[mask]
        if l.size < 100:
            raise ValueError(f"Insufficient tissue pixels: {l.size} < 100")

    l_mean, l_std = l.mean(), l.std()
    a_mean, a_std = a.mean(), a.std()
    b_mean, b_std = b.mean(), b.std()

    return (l_mean, a_mean, b_mean), (l_std, a_std, b_std)

def apply_reinhard(
    source_bgr: np.ndarray,
    target_means: Tuple[float, float, float],
    target_stds: Tuple[float, float, float],
    tissue_mask: Optional[np.ndarray] = None,
    bg_l_threshold: int = 230,
    min_saturation: int = 10
) -> np.ndarray:
    """
    Apply Reinhard stain normalization to tissue only; preserve white background.

    Args:
        source_bgr: Input BGR image
        target_means: Tuple (l_mean, a_mean, b_mean) from reference
        target_stds: Tuple (l_std, a_std, b_std) from reference
        tissue_mask: Pre-computed tissue mask (optional; computed if not provided)
        bg_l_threshold: L channel threshold for background detection (0–255)
        min_saturation: Minimum HSV saturation for tissue

    Returns:
        Normalized image with original background pixels restored.
    """
    if tissue_mask is None:
        tissue_mask = get_tissue_mask(source_bgr, bg_l_threshold, min_saturation)
    source_means, source_stds = get_lab_stats(source_bgr, mask=tissue_mask)

    lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    l = ((l - source_means[0]) * (target_stds[0] / (source_stds[0] + 1e-5))) + target_means[0]
    a = ((a - source_means[1]) * (target_stds[1] / (source_stds[1] + 1e-5))) + target_means[1]
    b = ((b - source_means[2]) * (target_stds[2] / (source_stds[2] + 1e-5))) + target_means[2]

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    lab_out = cv2.merge((l, a, b)).astype(np.uint8)
    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    result[~tissue_mask] = source_bgr[~tissue_mask]
    return result


def compute_template_from_tiles(tile_paths: list[str], bg_l_threshold: int = 230, min_saturation: int = 10) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute template statistics from a specific list of tile paths.

    Args:
        tile_paths: List of tile file paths
        bg_l_threshold: L channel threshold for background
        min_saturation: Minimum HSV saturation for tissue

    Returns:
        ((l_mean, a_mean, b_mean), (l_std, a_std, b_std)) — medians across samples
    """
    logger.info(f"Computing template from {len(tile_paths)} specific tiles...")

    all_means, all_stds = [], []
    for i, tile_path in enumerate(tile_paths, 1):
        img = leer_imagen(str(tile_path))
        if img is None:
            logger.warning(f"  Failed to read: {Path(tile_path).name}")
            continue

        mask = get_tissue_mask(img, bg_l_threshold, min_saturation)
        try:
            m, s = get_lab_stats(img, mask=mask)
            all_means.append(m)
            all_stds.append(s)
            logger.debug(f"  Tile {i}/{len(tile_paths)}: L={m[0]:.1f}±{s[0]:.1f}")
        except ValueError:
            logger.warning(f"  Insufficient tissue in: {Path(tile_path).name}")
            continue

    if len(all_means) < 5:
        raise ValueError(f"Only {len(all_means)} valid tiles found; need at least 5")

    target_means = tuple(np.median([m[i] for m in all_means]) for i in range(3))
    target_stds = tuple(np.median([s[i] for s in all_stds]) for i in range(3))

    logger.info(f"Template (median of {len(all_means)} tiles): L={target_means[0]:.1f}±{target_stds[0]:.1f}, a={target_means[1]:.1f}±{target_stds[1]:.1f}, b={target_means[2]:.1f}±{target_stds[2]:.1f}")
    return target_means, target_stds


def build_template_from_tiles(input_dir: str, n_samples: int = 200, bg_l_threshold: int = 230, min_saturation: int = 10) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Build robust template statistics from random sample of tiles.

    Args:
        input_dir: Directory to search for tiles (recursive)
        n_samples: Number of tiles to sample (default 200)
        bg_l_threshold: L channel threshold for background
        min_saturation: Minimum HSV saturation for tissue

    Returns:
        ((l_mean, a_mean, b_mean), (l_std, a_std, b_std)) — medians across samples

    Raises:
        ValueError: If insufficient valid tiles found.
    """
    input_path = Path(input_dir)
    candidates = [
        p for p in input_path.rglob('*')
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    ]

    if not candidates:
        raise ValueError(f"No tiles found in {input_dir}")

    chosen = random.sample(candidates, min(n_samples, len(candidates)))
    logger.info(f"Building template: sampling {len(chosen)} tiles from {input_dir}...")

    all_means, all_stds = [], []
    for i, tile_path in enumerate(chosen, 1):
        img = leer_imagen(str(tile_path))
        if img is None:
            continue

        mask = get_tissue_mask(img, bg_l_threshold, min_saturation)
        try:
            m, s = get_lab_stats(img, mask=mask)
            all_means.append(m)
            all_stds.append(s)
            if i % 50 == 0:
                logger.debug(f"  Sampled {i}/{len(chosen)} tiles")
        except ValueError:
            continue

    if len(all_means) < 10:
        raise ValueError(f"Only {len(all_means)} valid tiles found; need at least 10")

    target_means = tuple(np.median([m[i] for m in all_means]) for i in range(3))
    target_stds = tuple(np.median([s[i] for s in all_stds]) for i in range(3))

    logger.info(f"Template (median of {len(all_means)} tiles): L={target_means[0]:.1f}±{target_stds[0]:.1f}, a={target_means[1]:.1f}±{target_stds[1]:.1f}, b={target_means[2]:.1f}±{target_stds[2]:.1f}")
    return target_means, target_stds


def load_or_build_template(input_dir: str, template_stats_file: Optional[str] = None, rebuild: bool = False, bg_l_threshold: int = 230, min_saturation: int = 10) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Load cached template statistics or build from sample tiles.

    Args:
        input_dir: Directory with tiles (used to build template if needed)
        template_stats_file: Path to cached JSON file (default: .normalizacion/template_stats.json)
        rebuild: If True, force rebuild from tiles (ignore cache)
        bg_l_threshold: L channel threshold for background
        min_saturation: Minimum HSV saturation for tissue

    Returns:
        ((l_mean, a_mean, b_mean), (l_std, a_std, b_std))
    """
    if template_stats_file is None:
        template_stats_file = Path(input_dir).parent / ".normalizacion" / "template_stats.json"

    template_stats_file = Path(template_stats_file)

    if template_stats_file.exists() and not rebuild:
        try:
            with open(template_stats_file, 'r') as f:
                data = json.load(f)
                target_means = tuple(data['means'])
                target_stds = tuple(data['stds'])
                logger.info(f"Loaded cached template from {template_stats_file}")
                logger.info(f"Template (cached): L={target_means[0]:.1f}±{target_stds[0]:.1f}, a={target_means[1]:.1f}±{target_stds[1]:.1f}, b={target_means[2]:.1f}±{target_stds[2]:.1f}")
                return target_means, target_stds
        except Exception as e:
            logger.warning(f"Failed to load template cache: {e}; rebuilding...")

    # Build from tiles
    target_means, target_stds = build_template_from_tiles(input_dir, n_samples=200, bg_l_threshold=bg_l_threshold, min_saturation=min_saturation)

    # Cache for future runs
    try:
        template_stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(template_stats_file, 'w') as f:
            json.dump({
                'means': list(target_means),
                'stds': list(target_stds),
                'n_samples': 200
            }, f, indent=2)
        logger.info(f"Cached template to {template_stats_file}")
    except Exception as e:
        logger.warning(f"Could not cache template: {e}")

    return target_means, target_stds


def _normalize_single_tile(args: tuple) -> Optional[str]:
    """
    Normalize a single tile using Reinhard stain normalization.

    Args:
        args: Tuple of (source_path, output_path, target_means, target_stds,
                        min_tissue_ratio, bg_l_threshold, min_saturation)

    Returns:
        str: "success", "skipped", or None on error.
    """
    source_path, output_path, target_means, target_stds, min_tissue_ratio, bg_l_threshold, min_saturation = args

    try:
        img = leer_imagen(source_path)
        if img is None:
            logger.warning(f"Failed to read: {source_path}")
            return None

        tissue_mask = get_tissue_mask(img, bg_l_threshold, min_saturation)
        tissue_ratio = tissue_mask.sum() / tissue_mask.size

        if tissue_ratio < min_tissue_ratio:
            logger.debug(f"Skipped {Path(source_path).name}: tissue_ratio={tissue_ratio:.2%} < {min_tissue_ratio:.2%}")
            return "skipped"

        norm_img = apply_reinhard(img, target_means, target_stds, tissue_mask=tissue_mask, bg_l_threshold=bg_l_threshold, min_saturation=min_saturation)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        guardar_imagen(output_path, norm_img)
        return "success"

    except ValueError as e:
        logger.debug(f"Skipped {Path(source_path).name}: {e}")
        return "skipped"
    except Exception as e:
        logger.error(f"Error normalizing {source_path}: {e}", exc_info=True)
        return None


def _estimate_tile_memory_gb(tile_path: Path) -> float:
    """Estimate RAM needed for a tile: file_size × 8 (for uint8→float32 conversion + processing)."""
    if tile_path.exists():
        size_mb = tile_path.stat().st_size / (1024 ** 2)
        return (size_mb * 8) / 1024.0
    return 0.05  # Conservative default: 50 MB


def process_folder(input_dir: str, output_dir: str, template_path: Optional[str] = None, template_tiles: Optional[list[str]] = None, num_workers: Optional[int] = None, tile_size: int = 1024,
                   min_tissue_ratio: float = 0.05, bg_l_threshold: int = 230, min_saturation: int = 10, rebuild_template: bool = False,
                   ram_fraction: float = 0.80, min_free_gb: float = 2.0) -> None:
    """
    Recursively normalize tiles using dynamic parallelism (memory-aware admission control).

    Args:
        input_dir: Root directory (recursively searches for tiles)
        output_dir: Directory for output normalized tiles (mirrors folder structure)
        template_path: Path to reference template image (optional)
        template_tiles: List of specific tile paths to build template from (optional)
        num_workers: Number of parallel workers (None = auto-calculate based on RAM)
        tile_size: Tile size hint for RAM estimation (default 1024)
        min_tissue_ratio: Skip tiles with tissue < this ratio (default 0.05)
        bg_l_threshold: L channel threshold for background (default 230)
        min_saturation: Minimum HSV saturation for tissue (default 10)
        rebuild_template: If True, force rebuild template from tiles (ignore cache)
        ram_fraction: Fraction of available RAM to use for processing (0.9 default)
        min_free_gb: Minimum free RAM to always keep available (1.0 default)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading reference template...")
    try:
        if template_path and Path(template_path).is_file():
            logger.info(f"Using single template image: {template_path}")
            template_img = leer_imagen(template_path)
            if template_img is None:
                raise ValueError(f"Failed to read template image: {template_path}")
            mask = get_tissue_mask(template_img, bg_l_threshold, min_saturation)
            target_means, target_stds = get_lab_stats(template_img, mask=mask)
            logger.info(f"Template: L={target_means[0]:.1f}±{target_stds[0]:.1f}, a={target_means[1]:.1f}±{target_stds[1]:.1f}, b={target_means[2]:.1f}±{target_stds[2]:.1f}")
        elif template_tiles:
            target_means, target_stds = compute_template_from_tiles(template_tiles, bg_l_threshold=bg_l_threshold, min_saturation=min_saturation)
        else:
            target_means, target_stds = load_or_build_template(
                input_dir,
                template_stats_file=Path(input_dir).parent / ".normalizacion" / "template_stats.json",
                rebuild=rebuild_template,
                bg_l_threshold=bg_l_threshold,
                min_saturation=min_saturation
            )
    except ValueError as e:
        logger.error(f"Failed to load template: {e}")
        return

    valid_extensions = {'.png', '.jpg', '.jpeg'}
    input_path = Path(input_dir)

    tile_paths = [p for p in input_path.rglob('*') if p.suffix.lower() in valid_extensions]
    logger.info(f"Total de archivos encontrados: {len(tile_paths)}")

    # Separar imágenes de máscaras por convención de nombre
    image_paths = [p for p in tile_paths if not p.name.endswith('_mask.png')]
    mask_paths = [p for p in tile_paths if p.name.endswith('_mask.png')]
    logger.info(f"Imágenes a normalizar: {len(image_paths)}, Máscaras a copiar: {len(mask_paths)}")

    if len(image_paths) == 0:
        logger.warning("No se encontraron tiles PNG (excluidas máscaras). Nada que hacer.")
        return

    target_means, target_stds = load_or_build_template(
        input_dir,
        template_stats_file=Path(input_dir).parent / ".normalizacion" / "template_stats.json",
        rebuild=rebuild_template,
        bg_l_threshold=bg_l_threshold,
        min_saturation=min_saturation
    )

    # Sort by file size (descending) for load balancing
    image_paths.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)

    if num_workers is None:
        num_workers = _safe_worker_count(tile_size_hint=tile_size)

    # Build task list with output paths
    tasks = deque()
    task_memory_estimates: Dict = {}

    for tile_path in image_paths:
        rel_path = tile_path.relative_to(input_path)
        output_path = Path(output_dir) / rel_path
        task_tuple = (str(tile_path), str(output_path), target_means, target_stds, min_tissue_ratio, bg_l_threshold, min_saturation)
        mem_est = _estimate_tile_memory_gb(tile_path)
        tasks.append(task_tuple)
        task_memory_estimates[str(tile_path)] = mem_est

    # RAM budget (one-time calculation at startup)
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        available_gb = 8.0

    ram_budget_gb = available_gb * ram_fraction
    logger.info(f"RAM budget: {available_gb:.2f}GB available, using {ram_budget_gb:.2f}GB ({ram_fraction*100:.0f}%) with {min_free_gb}GB safety margin")
    logger.info(f"Starting dynamic parallel normalization with {num_workers} workers, {len(image_paths)} tiles\n")

    start_time = time.time()
    success_count = 0
    skip_count = 0
    failed_count = 0
    completed = 0

    # Dynamic admission loop with ProcessPoolExecutor
    used_ram_gb = 0.0
    active_futures: Dict = {}  # Maps Future → (tile_path_str, memory_estimate)

    def try_submit() -> None:
        """Try to submit next task from queue if RAM budget allows."""
        nonlocal used_ram_gb
        while tasks:
            task = tasks[0]
            tile_path_str = task[0]
            mem_est = task_memory_estimates[tile_path_str]

            # Dual RAM check: software counter AND live OS query
            try:
                import psutil
                os_available = psutil.virtual_memory().available / (1024 ** 3)
            except ImportError:
                os_available = ram_budget_gb + min_free_gb

            if used_ram_gb + mem_est <= ram_budget_gb and os_available > min_free_gb:
                # Submit task
                task = tasks.popleft()
                future = executor.submit(_normalize_single_tile, task)
                active_futures[future] = (task[0], mem_est)
                used_ram_gb += mem_est
            else:
                # Wait for running tasks to finish
                break

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        try_submit()

        while active_futures:
            done, _ = wait(active_futures, return_when=FIRST_COMPLETED)

            for future in done:
                tile_path_str, mem_est = active_futures.pop(future)
                used_ram_gb -= mem_est

                try:
                    result = future.result()
                    completed += 1
                    if result == "success":
                        success_count += 1
                    elif result == "skipped":
                        skip_count += 1
                    else:
                        failed_count += 1

                    if completed % 50 == 0:
                        logger.info(f"Tile {completed}/{len(tile_paths)} completed ({success_count} success, {skip_count} skipped, {failed_count} failed)")
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    failed_count += 1

                # Try to admit next task after one finishes
                try_submit()

    elapsed = time.time() - start_time
    logger.info(f"\n✓ Procesamiento de imágenes completado: {completed}/{len(image_paths)} tiles ")

    # Copiar máscaras sin transformar (son labels 0-4, no se normalizan)
    logger.info(f"Copiando {len(mask_paths)} máscaras sin transformación...")
    copied = 0
    failed = 0
    for mask_path in mask_paths:
        try:
            rel_path = mask_path.relative_to(input_dir)
            output_path = Path(output_dir) / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_path, output_path)
            copied += 1
        except Exception as e:
            logger.error(f"Error copiando {mask_path}: {e}")
            failed += 1

    logger.info(f"✓ Máscaras copiadas: {copied}/{len(mask_paths)} ({failed} fallos)")
    logger.info(f"✓ Procesamiento completado: {completed + copied}/{len(image_paths) + len(mask_paths)} archivos totales")

if __name__ == "__main__":
    import argparse

    # Defaults
    base_dir = Path("Salidas/Tiles_UNet")
    default_input = str(base_dir)
    default_output = str(Path("Salidas/Normalizados"))

    # Default template references — reads all images from referencias/ folder
    referencias_dir = Path("referencias")
    if referencias_dir.exists():
        default_template_tiles = sorted([
            str(p) for p in referencias_dir.glob('*')
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ])
    else:
        default_template_tiles = []

    parser = argparse.ArgumentParser(
        description="Normalización de Reinhard para tiles histopatológicos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uso por defecto (sin argumentos — usa 24 tiles representativos como template):
  python3 normalizacion.py

Rutas por defecto:
  --input    """ + default_input + """
  --output   """ + default_output + """

Uso personalizado:
  python3 normalizacion.py \\
    --input /ruta/a/tiles \\
    --output /ruta/salida \\
    --template /ruta/referencia.png \\
    --workers 4 \\
    --min_tissue_ratio 0.1
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
        default=None,
        help="Ruta a imagen de referencia única (opcional; por defecto: tiles representativos)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de workers paralelos (default: auto-calcula basado en RAM)"
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="Tamaño de tile para estimación de RAM (default: 1024)"
    )
    parser.add_argument(
        "--min_tissue_ratio",
        type=float,
        default=0.05,
        help="Omitir tiles con tejido < este ratio (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--bg_l_threshold",
        type=int,
        default=230,
        help="Umbral de canal L para detectar fondo blanco (0-255, default: 230)"
    )
    parser.add_argument(
        "--min_saturation",
        type=int,
        default=10,
        help="Saturación mínima HSV para detectar tejido (0-255, default: 10)"
    )
    parser.add_argument(
        "--rebuild-template",
        action="store_true",
        help="Recalcular template desde tiles (ignora cache)"
    )
    parser.add_argument(
        "--ram-fraction",
        type=float,
        default=0.9,
        help="Fracción de RAM disponible para usar (0.0-1.0, default: 0.9)"
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=1.0,
        help="Mínimo GB de RAM libre a mantener (default: 1.0)"
    )

    args = parser.parse_args()
    process_folder(
        args.input,
        args.output,
        template_path=args.template,
        template_tiles=default_template_tiles,
        num_workers=args.workers,
        tile_size=args.tile_size,
        min_tissue_ratio=args.min_tissue_ratio,
        bg_l_threshold=args.bg_l_threshold,
        min_saturation=args.min_saturation,
        rebuild_template=args.rebuild_template,
        ram_fraction=args.ram_fraction,
        min_free_gb=args.min_free_gb
    )