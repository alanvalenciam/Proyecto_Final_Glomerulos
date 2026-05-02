#!/usr/bin/env python3
"""
Tile Reconstruction Script
==========================

Reconstructs a full image from a set of PNG tiles organized by absolute coordinates.

Each tile is named following the pattern:
  {prefix}_tile_x{x}_y{y}_endx{endx}_endy{endy}.png

Where (x, y) and (endx, endy) define the absolute pixel positions in the canvas.

Usage:
  python reconstruct_tiles.py <input_folder> <output_path>

Example:
  python reconstruct_tiles.py \
    "/path/to/Salidas/Imagen/18-139" \
    "/path/to/tmp/18-139_layout.png"

Features:
  - Automatic canvas dimension detection from tile coordinates
  - RGB/RGBA support (converts to RGB if needed)
  - Progress logging and error handling
  - No heavy dependencies (uses PIL/Pillow only)
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from PIL import Image

# Increase limits for large images (metadata chunks + pixel count)
Image.MAX_IMAGE_PIXELS = 1_000_000_000  # 1 billion pixels
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100 MB for text chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_tile_coordinates(filename: str) -> Optional[Dict[str, int]]:
    """
    Extract (x, y, endx, endy) from tile filename.

    Args:
        filename: Tile filename, e.g., "18-139_tile_x10752_y21504_endx12288_endy23040.png"

    Returns:
        Dict with keys 'x', 'y', 'endx', 'endy', or None if parsing fails.
    """
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
    """
    Scan folder for PNG tiles and extract their coordinates.

    Args:
        folder: Path to folder containing tile PNGs

    Returns:
        List of (filename, coords_dict) tuples, sorted by (y, x)

    Raises:
        ValueError: If no PNG tiles are found in the folder
    """
    tiles = []

    for filename in os.listdir(folder):
        if not filename.endswith('.png'):
            continue

        coords = parse_tile_coordinates(filename)
        if coords is None:
            logger.warning(f"Skipping {filename}: Could not parse coordinates")
            continue

        tiles.append((filename, coords))

    if not tiles:
        raise ValueError(f"No valid PNG tiles found in {folder}")

    # Sort by y, then x for logical ordering
    tiles.sort(key=lambda t: (t[1]['y'], t[1]['x']))

    logger.info(f"Found {len(tiles)} valid tiles in {folder}")
    return tiles


def calculate_canvas_dimensions(tiles: List[Tuple[str, Dict[str, int]]], zoom_scale: float = 1.0) -> Tuple[int, int, int, int]:
    """
    Calculate canvas dimensions and offset from tile coordinates.

    Args:
        tiles: List of (filename, coords_dict) tuples
        zoom_scale: Zoom scale used during tiling (for dynamic sizing)

    Returns:
        Tuple of (canvas_width, canvas_height, min_x, min_y)
    """
    min_x = min(t[1]['x'] for t in tiles)
    max_x = max(t[1]['endx'] for t in tiles)
    min_y = min(t[1]['y'] for t in tiles)
    max_y = max(t[1]['endy'] for t in tiles)

    canvas_width = max_x - min_x
    canvas_height = max_y - min_y

    logger.info(f"Canvas dimensions: {canvas_width}x{canvas_height}")
    logger.info(f"X range: {min_x} to {max_x}")
    logger.info(f"Y range: {min_y} to {max_y}")
    if zoom_scale != 1.0:
        logger.info(f"Zoom scale detected: {zoom_scale}")

    return canvas_width, canvas_height, min_x, min_y


def reconstruct_image(
    folder: str,
    output_path: str,
    verbose: bool = True,
    zoom_scale: float = 0.5,
    max_size_mb: int = 50
) -> None:
    """
    Reconstruct full image from tiles.

    Args:
        folder: Path to folder containing tile PNGs
        output_path: Path where reconstructed image will be saved
        verbose: Print progress information
        zoom_scale: Zoom scale used during tiling (default 0.5, override with metadata.json if exists)
        max_size_mb: Target maximum file size in MB (default 50)

    Raises:
        FileNotFoundError: If folder does not exist
        ValueError: If no valid tiles are found
        IOError: If image cannot be saved
    """
    folder = os.path.abspath(folder)
    output_path = os.path.abspath(output_path)

    # Validate input
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Discover and parse tiles
    tiles = discover_tiles(folder)


    zoom_scale = 0.5

    canvas_w, canvas_h, min_x, min_y = calculate_canvas_dimensions(tiles, zoom_scale)

    # Create blank canvas (white background, RGB)
    logger.info(f"Creating canvas {canvas_w}x{canvas_h}...")
    canvas = Image.new('RGB', (canvas_w, canvas_h), color='white')

    # Paste each tile onto canvas
    logger.info("Pasting tiles onto canvas...")
    for i, (filename, coords) in enumerate(tiles, 1):
        tile_path = os.path.join(folder, filename)

        try:
            tile_img = Image.open(tile_path)

            # Convert RGBA to RGB if needed
            if tile_img.mode == 'RGBA':
                tile_img = tile_img.convert('RGB')
            elif tile_img.mode != 'RGB':
                tile_img = tile_img.convert('RGB')

            paste_x = coords['x'] - min_x
            paste_y = coords['y'] - min_y

            # Scale tile to match coordinate space
            # Coordinates are in original image space, tile pixels are in scaled space
            # If zoom was 0.5, tile needs to be scaled up 2x to fill the original coordinates
            scale_factor = 1.0 / zoom_scale if zoom_scale != 1.0 else 1.0
            if scale_factor != 1.0:
                new_width = int(tile_img.size[0] * scale_factor)
                new_height = int(tile_img.size[1] * scale_factor)
                tile_img = tile_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Paste tile (now properly sized for original coordinate space)
            canvas.paste(tile_img, (paste_x, paste_y))

            if verbose and i % 5 == 0:
                logger.info(f"  Processed {i}/{len(tiles)} tiles...")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            raise

    # Save result with compression and resizing to meet size target
    logger.info(f"Saving reconstructed image to {output_path}...")
    logger.info(f"Original canvas size: {canvas.size[0]}x{canvas.size[1]} pixels")

    # Determine output format and compression strategy
    output_ext = os.path.splitext(output_path)[1].lower()

    # Start with quality 85, adjust down if needed
    quality = 85
    max_dimension = max(canvas.size)

    # Estimate: reduce dimension if canvas is very large (>8000px)
    scale_factor = 1.0
    if max_dimension > 8000:
        scale_factor = 8000.0 / max_dimension
        logger.info(f"Downscaling image by {scale_factor:.2%} to reduce file size")
        new_size = (int(canvas.size[0] * scale_factor), int(canvas.size[1] * scale_factor))
        canvas = canvas.resize(new_size, Image.Resampling.LANCZOS)

    # Save as JPEG for strong compression
    save_format = 'JPEG'
    save_path = output_path.replace(output_ext, '.jpg') if output_ext != '.jpg' else output_path

    logger.info(f"Saving as JPEG with quality={quality}...")
    canvas.save(save_path, save_format, quality=quality, optimize=True)

    # Check file size and adjust quality if needed
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")

    # If still too large, reduce quality
    while file_size_mb > max_size_mb and quality > 50:
        quality -= 5
        logger.info(f"File too large, reducing quality to {quality}...")
        canvas.save(save_path, save_format, quality=quality, optimize=True)
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.1f} MB")

    logger.info(f"Success! Image saved to {save_path}")
    logger.info(f"Final image size: {canvas.size[0]}x{canvas.size[1]} pixels")
    logger.info(f"Final file size: {file_size_mb:.1f} MB")


def batch_reconstruct(parent_folder: str, output_folder: str, zoom_scale: float = 0.5) -> None:
    """
    Recursively reconstruct all images from tile subfolders.

    Args:
        parent_folder: Path to parent folder containing case subfolders
        output_folder: Path where all reconstructed images will be saved
        zoom_scale: Zoom scale used during tiling (default 0.5)
    """
    parent_folder = os.path.abspath(parent_folder)
    output_folder = os.path.abspath(output_folder)

    if not os.path.isdir(parent_folder):
        raise FileNotFoundError(f"Parent folder not found: {parent_folder}")

    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Processing all cases from: {parent_folder}")
    logger.info(f"Output folder: {output_folder}")

    # Collect all subfolders
    subfolders = [
        d for d in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, d)) and not d.startswith('.')
    ]
    subfolders.sort()

    logger.info(f"Found {len(subfolders)} cases to process")

    results = {'success': 0, 'failed': 0, 'skipped': 0}

    for i, case_name in enumerate(subfolders, 1):
        case_path = os.path.join(parent_folder, case_name)
        output_path = os.path.join(output_folder, f"{case_name}_reconstructed.png")

        # Check if folder has PNG tiles
        has_tiles = any(f.endswith('.png') for f in os.listdir(case_path))
        if not has_tiles:
            logger.warning(f"[{i}/{len(subfolders)}] SKIPPED {case_name}: No PNG tiles found")
            results['skipped'] += 1
            continue

        try:
            logger.info(f"[{i}/{len(subfolders)}] Processing {case_name}...")
            reconstruct_image(case_path, output_path, verbose=False, zoom_scale=zoom_scale, max_size_mb=50)
            results['success'] += 1
            logger.info(f"  ✓ {case_name} reconstructed successfully")
        except Exception as e:
            results['failed'] += 1
            logger.error(f"  ✗ {case_name} failed: {e}")

    # Summary
    logger.info("="*60)
    logger.info("RECONSTRUCTION SUMMARY")
    logger.info(f"  Success: {results['success']}")
    logger.info(f"  Failed: {results['failed']}")
    logger.info(f"  Skipped: {results['skipped']}")
    logger.info(f"  Total: {len(subfolders)}")
    logger.info(f"Output folder: {output_folder}")
    logger.info("="*60)


def main():
    """Main entry point for command-line usage."""
    # Default behavior: no arguments = batch mode on Salidas/Imagen
    if len(sys.argv) == 1:
        # Get the script directory and construct default paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_folder = os.path.join(script_dir, "Salidas", "Imagen")
        output_folder = os.path.join(script_dir, "tmp", "reconstructed")

        try:
            batch_reconstruct(input_folder, output_folder, zoom_scale=0.5)
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)
        return

    # Explicit argument mode
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage (default - auto batch):")
        print('  python reconstruct_tiles.py')
        print("  (processes all subfolders in ./Salidas/Imagen -> ./tmp/reconstructed/)")
        print("\nUsage (single case):")
        print('  python reconstruct_tiles.py <input_folder> <output_path> [zoom_scale]')
        print("\nUsage (batch - explicit paths):")
        print('  python reconstruct_tiles.py <parent_folder> <output_folder> --batch [zoom_scale]')
        print("\nExamples:")
        print('  # Auto batch (default, zoom=0.5):')
        print('  python reconstruct_tiles.py')
        print('\n  # Single case with default zoom:')
        print('  python reconstruct_tiles.py "/path/to/18-139" "/path/to/output.png"')
        print('\n  # Single case with custom zoom:')
        print('  python reconstruct_tiles.py "/path/to/18-139" "/path/to/output.png" 0.5')
        print('\n  # Batch with explicit paths:')
        print('  python reconstruct_tiles.py "/path/to/Salidas/Imagen" "/path/to/reconstructed" --batch')
        sys.exit(1)

    input_folder = sys.argv[1]
    output_path = sys.argv[2]
    batch_mode = '--batch' in sys.argv

    # Parse zoom_scale from remaining arguments
    zoom_scale = 0.5
    for arg in sys.argv[3:]:
        try:
            val = float(arg)
            if val > 0:
                zoom_scale = val
                break
        except (ValueError, IndexError):
            continue

    try:
        if batch_mode:
            batch_reconstruct(input_folder, output_path, zoom_scale=zoom_scale)
        else:
            reconstruct_image(input_folder, output_path, verbose=True, zoom_scale=zoom_scale, max_size_mb=50)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
