#!/usr/bin/env python3
"""
Reconstruct WSI from tiles in Salidas/Imagen/<slide>/images/ with class-colored mask overlay.

Structure:
    Salidas/Imagen/<slide>/
        ├─ images/
        │   ├─ <slide>_tile_x00000_y00000_endx01024_endy01024.png
        │   └─ ...
        ├─ masks/
        │   ├─ <slide>_tile_x00000_y00000_endx01024_endy01024_mask.png (grayscale encoded classes)
        │   └─ ...
        ├─ metadata.json
        └─ glomerulus_index.json

Mask encoding (grayscale values):
    - 0: background
    - 64: No_Proliferativo (shown as green)
    - 128: Proliferativo (shown as red)
    - 192: Esclerosado (shown as orange)
    - 255: Excluido (shown as purple)

Output:
    Salidas/Imagen/<slide>/reconstructions/
        ├─ <slide>_reconstruction.png     (tiles only)
        ├─ <slide>_mask.png               (masks only, grayscale with class values)
        └─ <slide>_overlay.png            (tiles + class-colored mask overlay)
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Gray value to class mapping
GRAY_VALUE_MAP = {
    0: 'background',
    64: 'No_Proliferativo',
    128: 'Proliferativo',
    192: 'Esclerosado',
    255: 'Excluido',
}


def parse_tile_filename(filename: str) -> Optional[Dict]:
    """Parse tile filename to extract native coordinates.

    Format: <slide>_tile_x{x:05d}_y{y:05d}_endx{x_end:05d}_endy{y_end:05d}.png
    """
    pattern = r'tile_x(\d+)_y(\d+)_endx(\d+)_endy(\d+)'
    match = re.search(pattern, filename)
    if not match:
        return None

    x, y, x_end, y_end = map(int, match.groups())
    return {'x': x, 'y': y, 'x_end': x_end, 'y_end': y_end}


def load_metadata(slide_dir: Path) -> Dict:
    """Load metadata.json to get native dimensions."""
    metadata_path = slide_dir / 'metadata.json'
    if not metadata_path.exists():
        logger.warning(f"metadata.json not found in {slide_dir}")
        return {}

    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load metadata: {e}")
        return {}


def collect_tiles(images_dir: Path) -> List[Tuple[str, Dict]]:
    """Collect all tile images and their coordinates."""
    tiles = []
    for image_path in sorted(images_dir.glob('*.png')):
        coords = parse_tile_filename(image_path.name)
        if coords:
            tiles.append((image_path.name, coords))

    return tiles


def gray_value_to_color(gray_value: int) -> Tuple[int, int, int]:
    """Map grayscale value to RGB color for visualization.

    Returns RGB tuple for each class:
    - 0 (background): (0, 0, 0) black - not shown
    - 64 (No_Proliferativo): (0, 255, 0) green
    - 128 (Proliferativo): (255, 0, 0) red
    - 192 (Esclerosado): (255, 165, 0) orange
    - 255 (Excluido): (128, 0, 128) purple
    """
    if gray_value == 0:
        return (0, 0, 0)  # background - black
    elif gray_value == 64:
        return (0, 255, 0)  # No_Proliferativo - green
    elif gray_value == 128:
        return (255, 0, 0)  # Proliferativo - red
    elif gray_value == 192:
        return (255, 165, 0)  # Esclerosado - orange
    elif gray_value == 255:
        return (128, 0, 128)  # Excluido - purple
    else:
        # For intermediate values, interpolate or default to gray
        return (gray_value, gray_value, gray_value)


def create_mask_overlay(image: Image.Image, mask: Image.Image, alpha: int = 180) -> Image.Image:
    """Composite class-colored mask onto RGB image with transparency.

    Each class gets a different color based on gray value:
    - 64 (No_Proliferativo): green
    - 128 (Proliferativo): red
    - 192 (Esclerosado): orange
    - 255 (Excluido): purple
    """
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.NEAREST)

    img_rgba = image.convert('RGBA')
    mask_arr = np.array(mask, dtype=np.uint8)

    # Create colored RGBA from mask
    h, w = mask_arr.shape
    rgba_arr = np.zeros((h, w, 4), dtype=np.uint8)

    # Apply class colors
    for gray_value in [64, 128, 192, 255]:
        mask_pixels = mask_arr == gray_value
        if np.any(mask_pixels):
            r, g, b = gray_value_to_color(gray_value)
            rgba_arr[mask_pixels, 0] = r
            rgba_arr[mask_pixels, 1] = g
            rgba_arr[mask_pixels, 2] = b
            rgba_arr[mask_pixels, 3] = alpha

    mask_rgba = Image.fromarray(rgba_arr, 'RGBA')
    overlay = Image.alpha_composite(img_rgba, mask_rgba)
    return overlay.convert('RGB')


def reconstruct_slide(
    slide_name: str,
    input_dir: Path = Path('Salidas/Imagen'),
    output_dir: Optional[Path] = None,
    scale: int = 10,
    overlay_alpha: int = 180,
    with_masks: bool = True,
) -> Tuple[bool, str]:
    """Reconstruct a single slide from tiles, accounting for zoom_scale."""
    if output_dir is None:
        output_dir = input_dir

    slide_dir = input_dir / slide_name
    images_dir = slide_dir / 'images'
    masks_dir = slide_dir / 'masks' if with_masks else None

    # Load metadata
    metadata = load_metadata(slide_dir)
    native_width = metadata.get('native_width', 0)
    native_height = metadata.get('native_height', 0)
    zoom_scale = metadata.get('zoom_scale', 1.0)  # Default to 1.0 if not present

    if native_width == 0 or native_height == 0:
        logger.warning(f"{slide_name}: Could not determine native dimensions from metadata")
        logger.warning(f"  metadata.json has: width={native_width}, height={native_height}")
        return False, f"Invalid native dimensions"

    # Collect tiles
    if not images_dir.exists():
        return False, f"images/ directory not found"

    tiles = collect_tiles(images_dir)
    if not tiles:
        return False, f"No tiles found in {images_dir}"

    logger.info(f"{slide_name}: Native {native_width}x{native_height}, zoom_scale={zoom_scale}, {len(tiles)} tiles")
    logger.info(f"  Class colors: 🟢=No_Prolif(64), 🔴=Prolif(128), 🟠=Esclero(192), 🟣=Excluido(255)")

    # Create canvas at native resolution, then downsample
    # Canvas is at native resolution scaled down by 'scale' factor
    canvas_width = max(1, native_width // scale)
    canvas_height = max(1, native_height // scale)

    wsi_img = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    mask_canvas = Image.new('L', (canvas_width, canvas_height), 0) if with_masks else None

    # Paste tiles
    loaded = 0
    failed = 0

    for tile_name, coords in tiles:
        image_path = images_dir / tile_name
        # Tile coordinates are in native space; scale them for the output canvas
        x, y = coords['x'] // scale, coords['y'] // scale

        if not image_path.exists():
            failed += 1
            continue

        try:
            tile_img = Image.open(image_path).convert('RGB')

            # Tile image is at output resolution (tile_size pixels)
            # Native tile size = tile_size / zoom_scale
            # Scaled size for canvas = native_tile_size / scale = (tile_size / zoom_scale) / scale
            native_tile_width = tile_img.width / zoom_scale
            native_tile_height = tile_img.height / zoom_scale
            w_s = max(1, int(native_tile_width / scale))
            h_s = max(1, int(native_tile_height / scale))

            tile_img_scaled = tile_img.resize((w_s, h_s), Image.LANCZOS)
            wsi_img.paste(tile_img_scaled, (x, y))
            loaded += 1

            # Load corresponding mask if exists
            if with_masks and masks_dir and masks_dir.exists():
                mask_path = masks_dir / tile_name.replace('.png', '_mask.png')
                if mask_path.exists():
                    try:
                        tile_mask = Image.open(mask_path).convert('L')
                        tile_mask_scaled = tile_mask.resize((w_s, h_s), Image.NEAREST)
                        mask_canvas.paste(tile_mask_scaled, (x, y))
                    except Exception as e:
                        logger.debug(f"  Could not load mask {mask_path.name}: {e}")

        except Exception as e:
            logger.debug(f"  Could not load tile {tile_name}: {e}")
            failed += 1

    if loaded == 0:
        return False, f"Failed to load any tiles"

    # Save outputs
    recon_dir = output_dir / slide_name / 'reconstructions'
    recon_dir.mkdir(parents=True, exist_ok=True)

    try:
        recon_path = recon_dir / f"{slide_name}_reconstruction.png"
        wsi_img.save(recon_path, 'PNG')

        if with_masks and mask_canvas:
            mask_path = recon_dir / f"{slide_name}_mask.png"
            mask_canvas.save(mask_path, 'PNG')

            overlay_path = recon_dir / f"{slide_name}_overlay.png"
            overlay = create_mask_overlay(wsi_img, mask_canvas, overlay_alpha)
            overlay.save(overlay_path, 'PNG')

            return True, f"✓ Loaded {loaded}/{loaded+failed} tiles → reconstruction + mask + overlay saved"

        return True, f"✓ Loaded {loaded}/{loaded+failed} tiles → reconstruction saved"

    except Exception as e:
        return False, f"Failed to save outputs: {e}"


def process_all_slides(
    input_dir: Path = Path('Salidas/Imagen'),
    output_dir: Optional[Path] = None,
    scale: int = 10,
    with_masks: bool = True,
    overlay_alpha: int = 180,
    max_workers: int = 4,
):
    """Process all slides in parallel."""
    if output_dir is None:
        output_dir = input_dir

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    slides = sorted([d.name for d in input_dir.iterdir() if d.is_dir()])
    if not slides:
        logger.info(f"No slides found in {input_dir}")
        return

    logger.info(f"Reconstructing {len(slides)} slides with {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(reconstruct_slide, slide, input_dir, output_dir, scale, overlay_alpha, with_masks): slide
            for slide in slides
        }

        completed = 0
        failed = 0

        with tqdm(total=len(slides), desc="Reconstructing", unit="slide") as pbar:
            for future in as_completed(futures):
                slide_name = futures[future]
                success, message = future.result()

                if success:
                    logger.info(f"  {message}")
                    completed += 1
                else:
                    logger.error(f"  ❌ {slide_name}: {message}")
                    failed += 1

                pbar.update(1)

    logger.info(f"\n✅ Complete: {completed} OK, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct WSI from tiles in Salidas/Imagen/<slide>/images/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # All slides with masks
  python reconstruct_imagen.py --all

  # Single slide
  python reconstruct_imagen.py --slide 29-10399

  # All slides, scale 5
  python reconstruct_imagen.py --all --scale 5

  # Without masks
  python reconstruct_imagen.py --all --no-masks
        """,
    )
    parser.add_argument(
        '--slide',
        help='Slide name (e.g., 29-10399); omit for --all',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('Salidas/Imagen'),
        help='Input directory (default: Salidas/Imagen)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: same as input-dir)',
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=10,
        help='Downsample factor (default: 10)',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all slides',
    )
    parser.add_argument(
        '--no-masks',
        action='store_true',
        help='Skip mask reconstruction',
    )
    parser.add_argument(
        '--overlay-alpha',
        type=int,
        default=180,
        help='Alpha of mask overlay (0-255, default: 180)',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)',
    )

    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir

    if args.all or not args.slide:
        process_all_slides(
            args.input_dir,
            output_dir,
            args.scale,
            not args.no_masks,
            args.overlay_alpha,
            args.workers,
        )
    else:
        if not args.slide:
            parser.print_help()
            return

        logger.info(f"Reconstructing {args.slide}...")
        success, message = reconstruct_slide(
            args.slide,
            args.input_dir,
            output_dir,
            args.scale,
            args.overlay_alpha,
            not args.no_masks,
        )
        if success:
            logger.info(f"  {message}")
        else:
            logger.error(f"  ❌ {message}")


if __name__ == '__main__':
    main()
