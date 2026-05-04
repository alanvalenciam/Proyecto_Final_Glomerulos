#!/usr/bin/env python3
"""
Convert all image formats to pyramidal TIFF using libvips.

Supports: .tif, .png, .jpg, .jpeg, .svs, .ndpi, .vms, .jp2
Output: Pyramidal TIFF files with multiple resolution levels.
"""

import subprocess
import os
import sys
import logging
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

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


SUPPORTED_FORMATS = {'.tif', '.png', '.jpg', '.jpeg', '.svs', '.ndpi', '.vms', '.jp2'}
SKIP_EXTENSIONS = {'.tiff', '.TIFF'}


def check_vips_installed():
    """Check if vips command is available."""
    try:
        subprocess.run(['vips', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def print_vips_installation_instructions():
    """Print instructions for installing libvips."""
    msg = "\n" + "=" * 70 + "\n"
    msg += "ERROR: 'vips' command not found\n"
    msg += "=" * 70 + "\n\n"
    msg += "To install libvips on macOS:\n"
    msg += "  brew install libvips\n\n"
    msg += "After installation, verify with:\n"
    msg += "  vips --version\n"
    msg += "=" * 70 + "\n"
    logger.error(msg)


def _safe_worker_count(image_paths: list) -> int:
    """
    Determine safe number of parallel workers based on available RAM.

    Strategy:
    - Get available RAM and reserve 30% (safety margin)
    - Estimate per-image RAM: max_file_size × 3 (vips processing overhead)
    - workers = min(cpu_count(), max(1, available_ram / max_image_ram))

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
            os.path.getsize(p) for p in image_paths
        ) if image_paths else 100 * 1024 * 1024

        estimated_per_image_ram = max_file_size * 3

        workers_by_ram = max(1, int(safe_ram / estimated_per_image_ram))
        workers = min(cpu_count(), workers_by_ram)

        logger.info(f"RAM-aware worker calculation: available={safe_ram / 1e9:.1f}GB, max_image={max_file_size / 1e6:.1f}MB, workers={workers}")
        return workers

    except Exception as e:
        logger.warning(f"psutil RAM calculation failed ({e}), using conservative worker count")
        return min(4, cpu_count())


def convert_to_pyramidal_tiff(input_path, output_path):
    """
    Convert a single image to pyramidal TIFF using vips.

    Args:
        input_path: Path to input image
        output_path: Path to output TIFF

    Returns:
        (success: bool, error_msg: str or None)
    """
    try:
        cmd = [
            'vips', 'tiffsave',
            str(input_path),
            str(output_path),
            '--tile',
            '--pyramid',
            '--compression=jpeg'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout per image
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, error_msg

        return True, None

    except subprocess.TimeoutExpired:
        return False, "Timeout (image too large or processing took >5min)"
    except Exception as e:
        return False, str(e)


def _process_single_image_to_tiff(args):
    """
    Worker function to convert a single image to pyramidal TIFF in parallel.

    Args:
        args: (file_path, verify)

    Returns:
        (filename, success: bool, error_msg: str or None, deleted: bool)
    """
    file_path, verify = args
    file_path = Path(file_path)
    filename = file_path.name

    try:
        # Skip if already TIFF
        if file_path.suffix.lower() in SKIP_EXTENSIONS:
            return (filename, True, None, False)

        output_path = get_output_path(file_path)

        success, error_msg = convert_to_pyramidal_tiff(str(file_path), str(output_path))

        if success:
            # Conversion succeeded - delete original (no verification in batch mode)
            deleted = False
            try:
                file_path.unlink()
                deleted = True
            except Exception:
                pass

            return (filename, True, None, deleted)
        else:
            return (filename, False, error_msg, False)

    except Exception as e:
        return (filename, False, str(e), False)


def should_convert(file_path):
    """Check if file should be converted based on extension."""
    suffix = file_path.suffix.lower()
    return suffix in SUPPORTED_FORMATS


def get_output_path(file_path):
    """Generate output TIFF path."""
    return file_path.with_suffix('.tiff')


def convert_all_to_pyramidal_tiff(input_dir, verify=True, num_workers=None):
    """
    Convert all supported image formats to pyramidal TIFF (parallelized).

    Args:
        input_dir: Directory containing images
        verify: If True, ask for confirmation before deleting original files (ignored in parallel mode)
        num_workers: Number of parallel workers. If None, auto-calculated based on available RAM.
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.error(f"Error: Directory not found: {input_dir}")
        return

    if not input_path.is_dir():
        logger.error(f"Error: Not a directory: {input_dir}")
        return

    # Check vips availability
    if not check_vips_installed():
        print_vips_installation_instructions()
        return

    # Find all supported image files
    image_files = []
    for suffix in SUPPORTED_FORMATS:
        image_files.extend(input_path.glob(f'*{suffix}'))
        image_files.extend(input_path.glob(f'*{suffix.upper()}'))

    # Remove duplicates and sort by size (largest first)
    image_files = sorted(set(image_files), key=lambda x: os.path.getsize(x), reverse=True)

    if not image_files:
        logger.info(f"No supported image files found in: {input_dir}")
        logger.info(f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
        return

    logger.info(f"Found {len(image_files)} image(s) to process")
    logger.info(f"Output directory: {input_path}\n")

    stats = {
        'total': len(image_files),
        'converted': 0,
        'skipped': 0,
        'failed': 0,
        'deleted': 0,
        'failed_files': []
    }

    # Determine number of workers
    if num_workers is None:
        num_workers = _safe_worker_count([str(f) for f in image_files])

    logger.info(f"Using {num_workers} parallel worker(s) for {len(image_files)} images\n")

    # Build task arguments (in parallel mode, verify is False = auto-delete)
    tasks = [(str(f), not verify) for f in image_files]

    start_time = time.time()

    # Process in parallel
    with Pool(num_workers) as pool:
        for idx, (filename, success, error_msg, deleted) in enumerate(pool.imap_unordered(_process_single_image_to_tiff, tasks), 1):
            # Skip if already TIFF
            if error_msg is None and not success and not deleted:
                logger.info(f"[{idx}/{len(image_files)}] ⏭️  {filename} (already .tiff)")
                stats['skipped'] += 1
                continue

            if success:
                if deleted:
                    logger.info(f"[{idx}/{len(image_files)}] ✅ {filename} (converted and deleted)")
                    stats['deleted'] += 1
                else:
                    logger.info(f"[{idx}/{len(image_files)}] ✅ {filename} (converted, original kept)")

                stats['converted'] += 1
            else:
                logger.error(f"[{idx}/{len(image_files)}] ❌ {filename}: {error_msg}")
                stats['failed'] += 1
                stats['failed_files'].append((filename, error_msg))

    elapsed = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total files:     {stats['total']}")
    logger.info(f"Converted:       {stats['converted']}")
    logger.info(f"Deleted:         {stats['deleted']}")
    logger.info(f"Skipped:         {stats['skipped']}")
    logger.info(f"Failed:          {stats['failed']}")
    logger.info(f"Time:            {elapsed:.1f}s")

    if stats['failed_files']:
        logger.info("\nFailed files:")
        for filename, error in stats['failed_files']:
            logger.error(f"  - {filename}: {error}")

    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    input_dir = r"/Users/olivera/Documents/Proyecto_Final_Glomerulos/Entradas"
    convert_all_to_pyramidal_tiff(input_dir, verify=False)
