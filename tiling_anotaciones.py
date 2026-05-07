#!/usr/bin/env python3
"""
Extract tiles ONLY from annotated polygons in GeoJSON files.
Enables reverse engineering by encoding bbox into filenames and saving manifest.json.

Pipeline:
1. For each TIFF+GeoJSON pair:
   - Load GeoJSON, filter out "Tissue" and "exclude" classifications
   - For each feature (polygon):
     * Compute bbox (shapely.bounds)
     * Add margin (relative or absolute)
     * Extract crop from TIFF (PIL or OpenSlide)
     * Resize to output_size x output_size
     * Save with encoded filename: {slide}_feat{idx:04d}_x{xmin:06d}_y{ymin:06d}_w{width:06d}_h{height:06d}.png
   - Save manifest.json per slide with original polygon coords and metadata

2. Output structure:
   Salidas/Tiles_Anotados/{slide_name}/
   ├── {tile_filename}.png
   └── manifest.json
"""

import os
import json
import logging
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from PIL import Image
import numpy as np
from shapely.geometry import shape, box

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    logger.warning("openslide-python not installed. SVS/NDPI/VMS formats will be skipped.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

OPENSLIDE_FORMATS = {'.svs', '.ndpi', '.vms'}
PIL_FORMATS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
LARGE_FILE_THRESHOLD = 300 * 1024 * 1024  # 300MB (use OpenSlide for larger files to avoid memory issues)


def calculate_safe_workers(tiff_paths: List[str], target_ram_percent: float = 0.5) -> int:
    """
    Calculate safe number of workers based on available RAM.

    Args:
        tiff_paths: list of TIFF file paths
        target_ram_percent: use up to this % of available RAM (default 0.9 = 90%)

    Returns:
        Number of workers (1-4)
    """
    if not PSUTIL_AVAILABLE:
        logger.info("psutil not available, using 2 workers by default")
        return min(2, cpu_count())

    try:
        # Get available RAM
        available_gb = psutil.virtual_memory().available / (1024**3)
        target_gb = available_gb * target_ram_percent

        # Estimate max file size
        max_file_gb = 0.0
        for path in tiff_paths:
            if os.path.exists(path):
                size_gb = os.path.getsize(path) / (1024**3)
                max_file_gb = max(max_file_gb, size_gb)

        if max_file_gb == 0:
            max_file_gb = 0.5

        # Each worker needs ~2x file size (decompressed TIFF, we resize early to minimize RAM)
        memory_per_worker = max_file_gb * 2.0

        if memory_per_worker > 0:
            # Use ceil for aggressive packing (round up if > 0.5)
            num_workers = math.ceil(target_gb / memory_per_worker)
            num_workers = max(1, min(num_workers, 4))  # Cap at 4
        else:
            num_workers = 2

        logger.info(
            f"RAM calculation: {available_gb:.1f}GB available, "
            f"max TIFF {max_file_gb:.2f}GB, "
            f"targeting {target_ram_percent*100:.0f}% → {num_workers} workers"
        )
        return num_workers

    except Exception as e:
        logger.warning(f"Failed to calculate workers: {e}, using 2 by default")
        return 2


def detect_format(file_path: str) -> Optional[str]:
    """Detect image format by extension. Always use PIL for large TIFF files (lazy loading)."""
    ext = os.path.splitext(file_path)[1].lower()

    # Try OpenSlide first for native WSI formats
    if ext in OPENSLIDE_FORMATS and OPENSLIDE_AVAILABLE:
        return 'openslide'
    # For TIFF/PNG/JPG, always use PIL (has lazy loading for large files)
    elif ext in PIL_FORMATS:
        return 'pil'
    return None


def load_image_metadata(image_path: str, fmt: str) -> Dict:
    """Load image dimensions and compute MPP info."""
    metadata = {'format': fmt}

    if fmt == 'openslide':
        with openslide.open_slide(image_path) as slide:
            w, h = slide.level_dimensions[0]
            metadata['width'] = w
            metadata['height'] = h
            try:
                mpp_x = float(slide.properties.get('openslide.mpp-x', 0.5))
                mpp_y = float(slide.properties.get('openslide.mpp-y', 0.5))
                metadata['mpp_x'] = mpp_x
                metadata['mpp_y'] = mpp_y
            except:
                metadata['mpp_x'] = 0.5
                metadata['mpp_y'] = 0.5
    else:  # PIL
        with Image.open(image_path) as img:
            metadata['width'], metadata['height'] = img.size
            metadata['mpp_x'] = 0.5  # Assume 0.5 microns per pixel for PIL
            metadata['mpp_y'] = 0.5

    return metadata


def extract_crop_from_tiff(
    image_path: str,
    bbox: Tuple[float, float, float, float],
    output_size: int,
    fmt: str,
    max_intermediate_size: int = 2048
) -> Optional[Image.Image]:
    """
    Extract crop from TIFF and resize to output_size x output_size.
    GeoJSON coordinates are always in level-0 (native) WSI space.

    Args:
        bbox: (xmin, ymin, xmax, ymax) in WSI level-0 pixel coordinates
        output_size: target output tile size in pixels
        fmt: 'pil' or 'openslide'
        max_intermediate_size: max size of crop before resize (default 2048, cap memory)

    Returns:
        PIL Image resized to (output_size, output_size) or None on error
    """
    try:
        xmin, ymin, xmax, ymax = bbox
        width = int(xmax - xmin)
        height = int(ymax - ymin)

        if width <= 0 or height <= 0:
            logger.debug(f"Invalid bbox dimensions: {width}x{height}")
            return None

        if fmt == 'openslide':
            with openslide.open_slide(image_path) as slide:
                region = slide.read_region(
                    (int(xmin), int(ymin)),
                    0,  # level 0 (native)
                    (int(width), int(height))
                )
                crop = region.convert('RGB')
        else:  # PIL for TIFF/PNG/JPG
            # For large TIFF files (>500MB), use OpenSlide to avoid memory issues
            file_size = os.path.getsize(image_path)
            is_large_tiff = (file_size > LARGE_FILE_THRESHOLD and
                           os.path.splitext(image_path)[1].lower() in {'.tif', '.tiff'})

            if is_large_tiff and OPENSLIDE_AVAILABLE:
                # Use OpenSlide for large TIFF files (more efficient)
                with openslide.open_slide(image_path) as slide:
                    region = slide.read_region(
                        (int(xmin), int(ymin)),
                        0,  # level 0 (native)
                        (int(width), int(height))
                    )
                    crop = region.convert('RGB')
            else:
                # Use PIL for smaller files or when OpenSlide is not available
                with Image.open(image_path) as img:
                    crop = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))

        if crop.size[0] <= 0 or crop.size[1] <= 0:
            logger.debug(f"Crop resulted in 0 size: {crop.size}")
            return None

        # Immediately resize if too large to avoid RAM explosion
        if crop.size[0] > max_intermediate_size or crop.size[1] > max_intermediate_size:
            logger.debug(f"Reducing large crop {crop.size} to intermediate size")
            # Maintain aspect ratio
            aspect = crop.size[0] / crop.size[1]
            if aspect > 1:
                new_w = max_intermediate_size
                new_h = int(max_intermediate_size / aspect)
            else:
                new_h = max_intermediate_size
                new_w = int(max_intermediate_size * aspect)
            crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Final resize to output_size (square)
        crop = crop.resize((output_size, output_size), Image.Resampling.LANCZOS)
        return crop

    except Exception as e:
        logger.error(f"Failed to extract crop from {image_path} at {bbox}: {e}")

    return None


def load_geojson(geojson_path: str) -> Dict:
    """Load GeoJSON file."""
    with open(geojson_path, 'r') as f:
        return json.load(f)


def process_slide_annotations(args: Dict) -> Dict:
    """
    Process a single TIFF+GeoJSON pair.

    Returns:
        {
            'slide_name': str,
            'image_path': str,
            'geojson_path': str,
            'tiles_extracted': int,
            'output_dir': str,
            'manifest': {...}
        }
    """
    slide_name = args['slide_name']
    image_path = args['image_path']
    geojson_path = args['geojson_path']
    output_dir = args['output_dir']
    output_size = args['output_size']
    margin_ratio = args['margin_ratio']

    slide_output_dir = os.path.join(output_dir, slide_name)
    os.makedirs(slide_output_dir, exist_ok=True)

    fmt = detect_format(image_path)
    if fmt is None:
        logger.error(f"Unsupported format for {image_path}")
        return {'slide_name': slide_name, 'tiles_extracted': 0, 'error': 'Unsupported format'}

    # Load GeoJSON first (avoid expensive image metadata load)
    try:
        geojson_obj = load_geojson(geojson_path)
    except Exception as e:
        logger.error(f"Failed to load GeoJSON {geojson_path}: {e}")
        return {'slide_name': slide_name, 'tiles_extracted': 0, 'error': f'GeoJSON error: {e}'}

    features = geojson_obj.get('features', [])

    # Process ALL features. Filter geometry type later.
    # (Include all annotations regardless of classification or properties)
    filtered_features = [(idx, f) for idx, f in enumerate(features)]

    # Lazy-load metadata. For large files, assume very large dimensions.
    img_width = 100000
    img_height = 100000

    manifest = {
        'slide': slide_name,
        'image_format': fmt,
        'image_path': image_path,
        'image_width': None,  # Will be filled lazily
        'image_height': None,
        'output_size': output_size,
        'margin_ratio': margin_ratio,
        'features': {}
    }

    tiles_extracted = 0

    for feat_idx, feature in filtered_features:
        try:
            geom = feature.get('geometry', {})
            if geom.get('type') != 'Polygon':
                logger.warning(f"Feature {feat_idx} in {slide_name} is not a Polygon, skipping")
                continue

            # Get bbox from shapely
            shp = shape(geom)
            xmin, ymin, xmax, ymax = shp.bounds

            # Apply margin
            width = xmax - xmin
            height = ymax - ymin
            max_dim = max(width, height)
            margin = max_dim * margin_ratio

            xmin_with_margin = max(0, xmin - margin)
            ymin_with_margin = max(0, ymin - margin)
            xmax_with_margin = min(img_width, xmax + margin)
            ymax_with_margin = min(img_height, ymax + margin)

            # Extract and resize
            bbox_with_margin = (xmin_with_margin, ymin_with_margin, xmax_with_margin, ymax_with_margin)
            crop = extract_crop_from_tiff(image_path, bbox_with_margin, output_size, fmt)

            if crop is None:
                logger.warning(f"Failed to extract crop for feature {feat_idx} in {slide_name}")
                continue

            # Generate filename
            w_wsi = int(xmax_with_margin - xmin_with_margin)
            h_wsi = int(ymax_with_margin - ymin_with_margin)
            tile_filename = (
                f"{slide_name}_feat{feat_idx:04d}_"
                f"x{int(xmin_with_margin):06d}_y{int(ymin_with_margin):06d}_"
                f"w{w_wsi:06d}_h{h_wsi:06d}.png"
            )

            tile_path = os.path.join(slide_output_dir, tile_filename)
            crop.save(tile_path, 'PNG')

            # Save to manifest
            manifest['features'][f"{feat_idx:04d}"] = {
                'tile_file': tile_filename,
                'original_polygon': geom['coordinates'][0],  # Ring
                'original_classification': feature.get('properties', {}).get('classification'),
                'original_name': feature.get('properties', {}).get('name'),
                'bbox_wsi': [float(xmin), float(ymin), float(xmax), float(ymax)],
                'bbox_with_margin': [float(xmin_with_margin), float(ymin_with_margin),
                                     float(xmax_with_margin), float(ymax_with_margin)]
            }

            tiles_extracted += 1
            logger.info(f"Extracted {tile_filename}")

        except Exception as e:
            logger.error(f"Error processing feature {feat_idx} in {slide_name}: {e}")
            continue

    # Save manifest
    manifest_path = os.path.join(slide_output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Completed {slide_name}: {tiles_extracted} tiles extracted")

    return {
        'slide_name': slide_name,
        'tiles_extracted': tiles_extracted,
        'output_dir': slide_output_dir,
        'manifest': manifest
    }


def find_tiff_geojson_pairs(input_dir: str) -> List[Dict]:
    """Find all TIFF+GeoJSON pairs in input_dir."""
    input_path = Path(input_dir)
    pairs = []

    # Find all TIFF files
    tiff_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))

    for tiff_path in sorted(tiff_files):
        stem = tiff_path.stem
        geojson_path = input_path / f"{stem}.geojson"

        if geojson_path.exists():
            pairs.append({
                'slide_name': stem,
                'image_path': str(tiff_path),
                'geojson_path': str(geojson_path)
            })
        else:
            logger.warning(f"No GeoJSON found for {stem}, skipping")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Extract tiles from annotated polygons in GeoJSON files.'
    )
    parser.add_argument('--input', type=str, default='Entradas',
                        help='Input directory with TIFF+GeoJSON pairs (default: Entradas)')
    parser.add_argument('--output', type=str, default='Salidas/Tiles_Anotados',
                        help='Output directory for tiles (default: Salidas/Tiles_Anotados)')
    parser.add_argument('--output_size', type=int, default=512,
                        help='Output tile size in pixels (square, default: 512)')
    parser.add_argument('--margin_ratio', type=float, default=0.2,
                        help='Margin as fraction of max bbox dimension (default: 0.2 = 20%%)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto-calculate based on RAM, max 4)')

    args = parser.parse_args()

    # Find pairs
    pairs = find_tiff_geojson_pairs(args.input)
    if not pairs:
        logger.error(f"No TIFF+GeoJSON pairs found in {args.input}")
        return

    logger.info(f"Found {len(pairs)} TIFF+GeoJSON pairs")

    # Prepare output dir
    os.makedirs(args.output, exist_ok=True)

    # Prepare worker args
    worker_args = []
    for pair in pairs:
        worker_args.append({
            'slide_name': pair['slide_name'],
            'image_path': pair['image_path'],
            'geojson_path': pair['geojson_path'],
            'output_dir': args.output,
            'output_size': args.output_size,
            'margin_ratio': args.margin_ratio
        })

    # Determine worker count
    if args.workers is not None:
        num_workers = args.workers
        logger.info(f"Using {num_workers} worker(s) (user specified)")
    else:
        # Calculate based on RAM (90% of available)
        tiff_paths = [p['image_path'] for p in pairs]
        num_workers = calculate_safe_workers(tiff_paths, target_ram_percent=0.5)
        logger.info(f"Auto-calculated: {num_workers} worker(s)")

    # Process in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_slide_annotations, worker_args)

    # Summary
    total_tiles = sum(r.get('tiles_extracted', 0) for r in results)
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY: {total_tiles} tiles extracted from {len(pairs)} slides")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
