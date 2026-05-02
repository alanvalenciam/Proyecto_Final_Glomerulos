#!/usr/bin/env python
"""
Glomeruli instance segmentation using torchvision's Mask R-CNN.
"""

import sys
import os
import json
import torch
import torchvision
import numpy as np
import cv2
from pathlib import Path
import tifffile
from collections import defaultdict

def get_wsi_info(image_path):
    """Get WSI metadata."""
    with tifffile.TiffFile(image_path) as tif:
        tags = tif.pages[0].tags
        width = tags.get('ImageWidth').value if 'ImageWidth' in tags else None
        height = tags.get('ImageLength').value if 'ImageLength' in tags else None
        return {
            'width': width,
            'height': height,
            'file_size_mb': os.path.getsize(image_path) / (1024*1024),
            'tile_size': (width, height),
        }

def load_model():
    """Load pretrained Mask R-CNN model."""
    print("Loading Mask R-CNN model from torchvision...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pretrained Mask R-CNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True,
        progress=True
    )
    model.to(device)
    model.eval()

    return model, device

def prepare_image(image_array, device):
    """Prepare image tensor for model."""
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
    img_tensor = img_tensor / 255.0  # Normalize to [0, 1]

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor.to(device)  # Return [C, H, W] without batch dimension

def tile_wsi(image_path, tile_size=1024, overlap_ratio=0.1):
    """
    Tile a large WSI image.

    Yields:
        (tile_array, metadata) tuples
    """
    print(f"\nTiling WSI with tile_size={tile_size}, overlap={overlap_ratio}")

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]

    stride = int(tile_size * (1 - overlap_ratio))
    tiles_generated = 0

    y_idx = 0
    y_pos = 0
    while y_pos < height:
        x_idx = 0
        x_pos = 0

        y_end = min(y_pos + tile_size, height)

        while x_pos < width:
            x_end = min(x_pos + tile_size, width)

            # Extract tile
            tile = img_rgb[y_pos:y_end, x_pos:x_end].copy()

            # If tile is smaller than tile_size, pad it
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            metadata = {
                'tile_idx_x': x_idx,
                'tile_idx_y': y_idx,
                'global_x': x_pos,
                'global_y': y_pos,
                'actual_width': x_end - x_pos,
                'actual_height': y_end - y_pos,
                'tile_size': tile_size,
            }

            tiles_generated += 1
            yield tile, metadata

            x_pos += stride
            x_idx += 1

        y_pos += stride
        y_idx += 1

    print(f"Generated {tiles_generated} tiles")

def process_tile(model, tile_array, device, score_threshold=0.5):
    """
    Process a single tile with Mask R-CNN.

    Returns:
        List of detections with masks
    """
    with torch.no_grad():
        img_tensor = prepare_image(tile_array, device)
        # Model expects list of [C, H, W] tensors
        predictions = model([img_tensor])

    result = predictions[0]
    boxes = result['boxes'].cpu().numpy()
    scores = result['scores'].cpu().numpy()
    masks = result['masks'].cpu().numpy()
    labels = result['labels'].cpu().numpy()

    # Filter by score threshold and by class (ignore background class)
    # For general purpose, we look for masks with high confidence
    detections = []
    for idx, score in enumerate(scores):
        if score > score_threshold:
            detection = {
                'box': boxes[idx].tolist(),
                'score': float(score),
                'label': int(labels[idx]),
                'mask': masks[idx][0]  # Remove channel dimension
            }
            detections.append(detection)

    return detections

def segment_wsi(image_path, output_dir, tile_size=1024, score_threshold=0.5):
    """
    Main WSI segmentation pipeline.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"WSI Glomeruli Segmentation Pipeline")
    print(f"{'='*70}")
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")

    # Get WSI info
    wsi_info = get_wsi_info(image_path)
    print(f"\nWSI Info:")
    print(f"  Size: {wsi_info['width']}x{wsi_info['height']} pixels")
    print(f"  File size: {wsi_info['file_size_mb']:.2f} MB")

    # Load model
    model, device = load_model()
    print(f"Model loaded successfully")

    # Collect all detections
    all_detections = defaultdict(list)
    tile_results = []

    # Process tiles
    print(f"\nProcessing tiles...")
    for tile_idx, (tile_array, metadata) in enumerate(tile_wsi(image_path, tile_size)):
        if (tile_idx + 1) % 10 == 0:
            print(f"  Processed {tile_idx + 1} tiles...")

        detections = process_tile(model, tile_array, device, score_threshold)

        # Adjust coordinates to global image space
        for det in detections:
            # Adjust box coordinates
            box = det['box']
            box[0] += metadata['global_x']
            box[1] += metadata['global_y']
            box[2] += metadata['global_x']
            box[3] += metadata['global_y']

            det['global_box'] = box
            det['tile_metadata'] = metadata

            all_detections[f"tile_{metadata['tile_idx_y']}_{metadata['tile_idx_x']}"].append(det)

        tile_results.append({
            'tile_idx': tile_idx,
            'metadata': metadata,
            'detection_count': len(detections),
        })

    # Summary statistics
    total_detections = sum(len(dets) for dets in all_detections.values())

    print(f"\n{'='*70}")
    print(f"Segmentation Results:")
    print(f"{'='*70}")
    print(f"Total tiles processed: {len(tile_results)}")
    print(f"Total detections (objects): {total_detections}")

    # Save results
    results_json = {
        'wsi_info': wsi_info,
        'processing_config': {
            'tile_size': tile_size,
            'score_threshold': score_threshold,
        },
        'statistics': {
            'total_tiles': len(tile_results),
            'total_detections': total_detections,
            'avg_detections_per_tile': total_detections / max(1, len(tile_results)),
        },
        'tile_results': tile_results,
    }

    results_file = os.path.join(output_dir, 'segmentation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Create summary image
    img_cv = cv2.imread(image_path)
    if img_cv is not None:
        # Create a smaller version for visualization
        scale = min(1.0, 4000 / max(img_cv.shape[:2]))
        h, w = int(img_cv.shape[0] * scale), int(img_cv.shape[1] * scale)
        img_small = cv2.resize(img_cv, (w, h))

        # Draw detections on summary
        for detections in all_detections.values():
            for det in detections:
                if len(detections) > 0:  # Only if we have detections
                    box = det['global_box']
                    # Scale box to summary image
                    x1, y1, x2, y2 = [int(c * scale) for c in box]
                    cv2.rectangle(img_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_small, f"{det['score']:.2f}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        summary_path = os.path.join(output_dir, 'segmentation_summary.jpg')
        cv2.imwrite(summary_path, img_small)
        print(f"Summary image saved to: {summary_path}")

    return results_json

def main():
    image_path = '/Users/olivera/Documents/Proyecto_Final_Glomerulos/Entradas/933-10155.tiff'
    output_dir = '/Users/olivera/Documents/Proyecto_Final_Glomerulos/Salidas'

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return False

    try:
        results = segment_wsi(image_path, output_dir, tile_size=512, score_threshold=0.5)
        print(f"\n{'='*70}")
        print(f"Segmentation completed successfully!")
        print(f"{'='*70}\n")
        return True
    except Exception as e:
        print(f"\nError during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
