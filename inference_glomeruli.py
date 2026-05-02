#!/usr/bin/env python
"""
Glomeruli instance segmentation inference script.
Processes WSI images to detect and segment glomeruli using Cascade Mask R-CNN.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import tifffile

# Add repo to path
sys.path.insert(0, '/Users/olivera/Documents/Proyecto_Final_Glomerulos/glomeruli-repo/code')

# Try to import mmdet components
try:
    from tools.mmdet.apis.inference import init_detector, inference_detector
    mmdet_available = True
except ImportError as e:
    print(f"Warning: mmdet not fully available: {e}")
    mmdet_available = False

def get_image_size(image_path):
    """Get WSI image dimensions without loading entire image."""
    with tifffile.TiffFile(image_path) as tif:
        tags = tif.pages[0].tags
        width = tags.get('ImageWidth').value if 'ImageWidth' in tags else None
        height = tags.get('ImageLength').value if 'ImageLength' in tags else None
        if width and height:
            return (width, height)
    # Fallback
    img = cv2.imread(image_path)
    if img is not None:
        return (img.shape[1], img.shape[0])
    return (0, 0)

def tile_wsi(image_path, tile_size=1024, overlap=0.1):
    """
    Tile a WSI image for processing.

    Args:
        image_path: Path to WSI TIFF file
        tile_size: Size of each tile
        overlap: Overlap ratio between tiles

    Yields:
        (tile, coords) tuples
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    height, width = img.shape[:2]
    stride = int(tile_size * (1 - overlap))

    print(f"WSI dimensions: {width}x{height}")
    print(f"Tile size: {tile_size}, Stride: {stride}")

    tile_count = 0
    y_start = 0
    while y_start < height:
        y_end = min(y_start + tile_size, height)
        x_start = 0
        while x_start < width:
            x_end = min(x_start + tile_size, width)

            # Extract tile
            tile = img[y_start:y_end, x_start:x_end].copy()

            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile

            coords = {
                'x': x_start,
                'y': y_start,
                'width': x_end - x_start,
                'height': y_end - y_start,
                'original_width': x_end - x_start,
                'original_height': y_end - y_start
            }

            tile_count += 1
            yield tile, coords

            x_start += stride

        y_start += stride

    print(f"Total tiles generated: {tile_count}")

def process_wsi(image_path, output_dir):
    """
    Process WSI image for glomeruli segmentation.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    # Get image info
    width, height = get_image_size(image_path)
    print(f"WSI size: {width}x{height} pixels")

    # For now, just analyze the image and create a basic report
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image")
        return False

    # Convert to RGB for analysis
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Analyze image
    print(f"\nImage shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min/max values: {img.min()}/{img.max()}")

    # Create a simple summary
    info = {
        'input_image': image_path,
        'width': int(width),
        'height': int(height),
        'file_size_mb': os.path.getsize(image_path) / (1024*1024),
        'status': 'analyzed',
        'note': 'WSI loaded and analyzed successfully. Ready for segmentation.'
    }

    # Save info
    info_file = os.path.join(output_dir, 'analysis.json')
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nAnalysis saved to: {info_file}")

    # Save thumbnail
    thumbnail_size = (512, 512)
    img_thumb = cv2.resize(img_rgb, thumbnail_size)
    thumb_path = os.path.join(output_dir, 'thumbnail.jpg')
    cv2.imwrite(thumb_path, cv2.cvtColor(img_thumb, cv2.COLOR_RGB2BGR))
    print(f"Thumbnail saved to: {thumb_path}")

    # Attempt model loading if mmdet is available
    if mmdet_available:
        print("\nAttempting to load Cascade Mask R-CNN model...")
        try:
            checkpoint_path = '/Users/olivera/Documents/Proyecto_Final_Glomerulos/models/Cascade_Mask-RCNN_snapshot.pth'
            config_path = '/Users/olivera/Documents/Proyecto_Final_Glomerulos/glomeruli-repo/code/tools/cascade_mask_rcnn_r50_fpn_1x.py'

            if os.path.exists(checkpoint_path):
                print(f"Model checkpoint found: {checkpoint_path}")
                print(f"Model config: {config_path}")
                # Model loading will be done in next phase
            else:
                print(f"Model checkpoint not found at {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("\nNote: Full mmdet installation needed for segmentation.")
        print("Current phase: Image analysis and validation")

    return True

def main():
    image_path = '/Users/olivera/Documents/Proyecto_Final_Glomerulos/Entradas/933-10155.tiff'
    output_dir = '/Users/olivera/Documents/Proyecto_Final_Glomerulos/Salidas'

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    success = process_wsi(image_path, output_dir)

    if success:
        print(f"\n{'='*60}")
        print("Phase 1 completed: Image analysis")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
