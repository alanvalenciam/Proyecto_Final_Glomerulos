#!/usr/bin/env python3
"""
Tile Reconstruction Script (MPP-Aware)
========================================

Reconstructs a full image from a set of tiles organized by native coordinates.

Each tile is named following the pattern:
  {prefix}_tile_x{x}_y{y}_endx{endx}_endy{endy}.png

Where (x, y) and (endx, endy) define absolute pixel positions in the NATIVE (level-0) coordinate space.

Tiles are at a target resolution (target_mpp). The script reads metadata.json to:
  - native_mpp: Original slide resolution
  - target_mpp: Tile export resolution
  - downsample_ratio: target_mpp / native_mpp

Usage:
  python reconstruct_tiles.py <input_folder> <output_path>

Features:
  - Automatic MPP detection from metadata.json
  - Correct coordinate mapping (native → output space)
  - No tile upscaling (tiles already at correct resolution)
"""

import os
import re
import sys
import json
import logging
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1_000_000_000
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

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


def parse_tile_coordinates(filename: str) -> Optional[Dict[str, int]]:
    """Extract (x, y, endx, endy) from tile filename."""
    match = re.search(r'x(\d+)_y(\d+)_endx(\d+)_endy(\d+)', filename)
    if not match:
        return None

    x, y, endx, endy = map(int, match.groups())
    return {
        'x': x,
        'y': y,
        'endx': endx,
        'endy': endy,
        'width': endx - x,
        'height': endy - y,
    }


def discover_tiles(folder: str) -> List[Tuple[str, Dict[str, int]]]:
    """Scan folder for tiles and extract their coordinates."""
    tiles = []

    for filename in os.listdir(folder):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue

        coords = parse_tile_coordinates(filename)
        if coords is None:
            logger.warning(f"Skipping {filename}: Could not parse coordinates")
            continue

        tiles.append((filename, coords))

    if not tiles:
        raise ValueError(f"No valid tiles found in {folder}")

    tiles.sort(key=lambda t: (t[1]['y'], t[1]['x']))
    logger.info(f"Found {len(tiles)} valid tiles in {folder}")
    return tiles


def load_metadata(folder: str) -> Dict:
    """Load metadata.json from tile folder."""
    metadata_path = os.path.join(folder, "metadata.json")

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata: native_mpp={metadata.get('native_mpp')}, target_mpp={metadata.get('target_mpp')}, downsample_ratio={metadata.get('downsample_ratio')}")
            return metadata
        except Exception as e:
            logger.warning(f"Could not load metadata.json: {e}")

    # Fallback for old-style tiles (without metadata)
    logger.warning("No metadata.json found. Using fallback scale=1.0")
    return {
        "native_mpp": None,
        "target_mpp": None,
        "downsample_ratio": 1.0
    }


def _safe_worker_count(tile_folders: List[str]) -> int:
    """
    Determine safe number of parallel workers based on available RAM.

    Strategy:
    - Get available RAM and reserve 30% (safety margin)
    - Estimate per-folder RAM: median_tile_size × max_tiles_in_folder × 1.5
    - workers = min(cpu_count(), max(1, available_ram / max_folder_ram))

    Returns:
        int: Number of workers to use in the pool (minimum 1)
    """
    try:
        if PSUTIL_AVAILABLE:
            available_ram = psutil.virtual_memory().available
        else:
            available_ram = 4 * 1024 * 1024 * 1024  # Conservative fallback: 4GB

        safe_ram = available_ram * 0.7

        # Estimate max per-folder size
        max_folder_size = 0
        for folder in tile_folders:
            if os.path.isdir(folder):
                folder_size = sum(
                    os.path.getsize(os.path.join(folder, f))
                    for f in os.listdir(folder)
                    if f.endswith(('.png', '.jpg', '.jpeg'))
                )
                max_folder_size = max(max_folder_size, folder_size)

        if max_folder_size == 0:
            max_folder_size = 100 * 1024 * 1024  # Conservative: 100MB

        estimated_per_folder_ram = max_folder_size * 1.5

        workers_by_ram = max(1, int(safe_ram / estimated_per_folder_ram))
        workers = min(cpu_count(), workers_by_ram)

        logger.info(f"RAM-aware worker calculation: available={safe_ram / 1e9:.1f}GB, max_folder={max_folder_size / 1e9:.1f}GB, workers={workers}")
        return workers

    except Exception as e:
        logger.warning(f"psutil RAM calculation failed ({e}), using conservative worker count")
        return min(4, cpu_count())


def calculate_canvas_dimensions(tiles: List[Tuple[str, Dict[str, int]]], downsample_ratio: float, image_format: str = 'pil') -> Tuple[int, int, int, int]:
    """Calculate canvas dimensions in target-mpp space."""
    min_x = min(t[1]['x'] for t in tiles)
    max_x = max(t[1]['endx'] for t in tiles)
    min_y = min(t[1]['y'] for t in tiles)
    max_y = max(t[1]['endy'] for t in tiles)

    # Native coordinates range
    native_width = max_x - min_x
    native_height = max_y - min_y

    # Scale to output space depends on format:
    # - OpenSlide: scale UP (1.0/downsample_ratio) because native → output
    # - PIL: scale DOWN (downsample_ratio) because native coords need to compress to output tile positions
    if image_format == 'openslide':
        scale = 1.0 / downsample_ratio
    else:  # PIL
        scale = downsample_ratio

    canvas_width = int(native_width * scale)
    canvas_height = int(native_height * scale)

    logger.info(f"Native coordinate range: {native_width}x{native_height} px")
    logger.info(f"Downsample ratio: {downsample_ratio:.4f}")
    logger.info(f"Image format: {image_format}")
    logger.info(f"Canvas dimensions (target space): {canvas_width}x{canvas_height} px")
    logger.info(f"X range (native): {min_x} to {max_x}")
    logger.info(f"Y range (native): {min_y} to {max_y}")

    return canvas_width, canvas_height, min_x, min_y


def _reconstruct_single_case(args) -> Tuple[str, bool, Optional[str]]:
    """
    Worker function to reconstruct a single case in parallel.

    Args:
        args: (case_path, output_path, verbose, max_size_mb)

    Returns:
        (case_name, success: bool, error_msg: Optional[str])
    """
    case_path, output_path, verbose, max_size_mb = args
    case_name = os.path.basename(case_path)

    try:
        reconstruct_image_internal(case_path, output_path, verbose, max_size_mb)
        return (case_name, True, None)
    except Exception as e:
        return (case_name, False, str(e))


def reconstruct_image_internal(
    folder: str,
    output_path: str,
    verbose: bool = True,
    max_size_mb: int = 50
) -> None:
    """Internal function that reconstructs full image from tiles."""
    folder = os.path.abspath(folder)
    output_path = os.path.abspath(output_path)

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Load metadata to get correct scaling
    metadata = load_metadata(folder)
    downsample_ratio = metadata.get('downsample_ratio', 1.0)
    image_format = metadata.get('format', 'pil')

    # Scale calculation depends on format:
    # - OpenSlide: tiles are in native space, need to downscale to output
    # - PIL: tiles already resized to output, coordinates are native, need to compress native→output
    if image_format == 'openslide':
        scale = 1.0 / downsample_ratio if downsample_ratio else 1.0
    else:  # PIL format
        scale = downsample_ratio

    # Discover tiles
    tiles = discover_tiles(folder)

    # Calculate canvas in target-mpp space
    canvas_w, canvas_h, min_x, min_y = calculate_canvas_dimensions(tiles, downsample_ratio, image_format=image_format)

    # Create canvas
    logger.info(f"Creating canvas {canvas_w}x{canvas_h}...")
    canvas = Image.new('RGB', (canvas_w, canvas_h), color='white')

    # Paste tiles
    logger.info("Pasting tiles onto canvas...")
    for i, (filename, coords) in enumerate(tiles, 1):
        tile_path = os.path.join(folder, filename)

        try:
            tile_img = Image.open(tile_path)

            # Convert to RGB if needed
            if tile_img.mode == 'RGBA':
                tile_img = tile_img.convert('RGB')
            elif tile_img.mode != 'RGB':
                tile_img = tile_img.convert('RGB')

            # Calculate paste position in target-mpp space
            paste_x = int((coords['x'] - min_x) * scale)
            paste_y = int((coords['y'] - min_y) * scale)

            # Tiles are already at correct pixel size (tile_size × tile_size in target_mpp)
            # No resizing needed
            canvas.paste(tile_img, (paste_x, paste_y))

            if verbose and i % 5 == 0:
                logger.info(f"  Processed {i}/{len(tiles)} tiles...")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            raise

    # Save result with compression
    logger.info(f"Saving reconstructed image to {output_path}...")
    logger.info(f"Canvas size: {canvas.size[0]}x{canvas.size[1]} pixels")

    output_ext = os.path.splitext(output_path)[1].lower()
    quality = 85
    max_dimension = max(canvas.size)

    # Estimate: reduce if very large
    scale_factor = 1.0
    if max_dimension > 8000:
        scale_factor = 8000.0 / max_dimension
        logger.info(f"Downscaling image by {scale_factor:.2%} to reduce file size")
        new_size = (int(canvas.size[0] * scale_factor), int(canvas.size[1] * scale_factor))
        canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)

    # Save as JPEG
    save_format = 'JPEG'
    save_path = output_path.replace(output_ext, '.jpg') if output_ext != '.jpg' else output_path

    logger.info(f"Saving as JPEG with quality={quality}...")
    canvas.save(save_path, save_format, quality=quality, optimize=True)

    # Adjust quality if needed
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")

    while file_size_mb > max_size_mb and quality > 50:
        quality -= 5
        logger.info(f"File too large, reducing quality to {quality}...")
        canvas.save(save_path, save_format, quality=quality, optimize=True)
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.1f} MB")

    logger.info(f"Success! Image saved to {save_path}")
    logger.info(f"Final image size: {canvas.size[0]}x{canvas.size[1]} pixels")
    logger.info(f"Final file size: {file_size_mb:.1f} MB")


def batch_reconstruct(parent_folder: str, output_folder: str, num_workers: Optional[int] = None) -> None:
    """Recursively reconstruct all images from tile subfolders (parallelized)."""
    parent_folder = os.path.abspath(parent_folder)
    output_folder = os.path.abspath(output_folder)

    if not os.path.isdir(parent_folder):
        raise FileNotFoundError(f"Parent folder not found: {parent_folder}")

    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Processing all cases from: {parent_folder}")
    logger.info(f"Output folder: {output_folder}")

    subfolders = [
        d for d in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, d)) and not d.startswith('.')
    ]
    subfolders.sort()

    logger.info(f"Found {len(subfolders)} cases to process")

    # Filter folders with tiles and sort by size (largest first)
    case_paths = [os.path.join(parent_folder, case) for case in subfolders]
    cases_with_tiles = []
    skipped_count = 0

    for case_path, case_name in zip(case_paths, subfolders):
        has_tiles = any(f.endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(case_path))
        if not has_tiles:
            logger.info(f"SKIPPED {case_name}: No tiles found")
            skipped_count += 1
        else:
            cases_with_tiles.append((case_path, case_name))

    # Sort by folder size (largest first)
    cases_with_tiles.sort(
        key=lambda x: sum(os.path.getsize(os.path.join(x[0], f)) for f in os.listdir(x[0])),
        reverse=True
    )

    if not cases_with_tiles:
        logger.info(f"No cases with tiles found")
        return

    logger.info(f"Found {len(cases_with_tiles)} cases with tiles, {skipped_count} skipped")

    # Determine number of workers
    if num_workers is None:
        num_workers = _safe_worker_count([case[0] for case in cases_with_tiles])

    logger.info(f"Using {num_workers} parallel worker(s) for {len(cases_with_tiles)} cases\n")

    # Build task arguments
    tasks = [
        (case_path, os.path.join(output_folder, f"{case_name}_reconstructed.jpg"), False, 50)
        for case_path, case_name in cases_with_tiles
    ]

    results = {'success': 0, 'failed': 0, 'skipped': skipped_count}
    start_time = time.time()

    # Process in parallel
    with Pool(num_workers) as pool:
        for idx, (case_name, success, error_msg) in enumerate(pool.imap_unordered(_reconstruct_single_case, tasks), 1):
            if success:
                results['success'] += 1
                logger.info(f"[{idx}/{len(cases_with_tiles)}] ✓ {case_name} reconstructed")
            else:
                results['failed'] += 1
                logger.error(f"[{idx}/{len(cases_with_tiles)}] ✗ {case_name} failed: {error_msg}")

    elapsed = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info("RECONSTRUCTION SUMMARY")
    logger.info(f"  Success: {results['success']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Skipped: {results['skipped']}")
    logger.info(f"  Total: {len(subfolders)}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info(f"Output folder: {output_folder}")
    logger.info("="*60)


def main():
    """Main entry point."""
    if len(sys.argv) == 1:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_folder = os.path.join(script_dir, "Salidas", "Imagen")
        output_folder = os.path.join(script_dir, "tmp", "reconstructed")

        try:
            batch_reconstruct(input_folder, output_folder)
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)
        return

    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage:")
        print('  python reconstruct_tiles.py <input_folder> <output_path>')
        print('  python reconstruct_tiles.py <parent_folder> <output_folder> --batch')
        print("\nExamples:")
        print('  python reconstruct_tiles.py')
        print('  python reconstruct_tiles.py "/path/to/tiles" "/path/to/output.jpg"')
        print('  python reconstruct_tiles.py "/path/to/Salidas/Imagen" "/path/to/reconstructed" --batch')
        sys.exit(1)

    input_folder = sys.argv[1]
    output_path = sys.argv[2]
    batch_mode = '--batch' in sys.argv

    try:
        if batch_mode:
            batch_reconstruct(input_folder, output_path)
        else:
            reconstruct_image(input_folder, output_path, verbose=True, max_size_mb=50)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
