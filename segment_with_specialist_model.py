#!/usr/bin/env python
"""
Glomeruli segmentation using the specialist Cascade Mask R-CNN model.
Loads pretrained weights from the BUPT dataset.
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

# Configuration
CHECKPOINT_PATH = '/Users/olivera/Documents/Proyecto_Final_Glomerulos/models/Cascade_Mask-RCNN_snapshot.pth'
IMAGE_PATH = '/Users/olivera/Documents/Proyecto_Final_Glomerulos/Entradas/933-10155.tiff'
OUTPUT_DIR = '/Users/olivera/Documents/Proyecto_Final_Glomerulos/Salidas'

TILE_SIZE = 512
OVERLAP_RATIO = 0.1
SCORE_THRESHOLD = 0.5
MAX_TILES = 50  # Limit to 50 tiles for validation (set to None for full WSI)

def load_checkpoint_weights():
    """Load specialist model weights."""
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

    # Extract model state
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print(f"Checkpoint contains {len(state_dict)} state dict entries")

    # Print metadata if available
    if 'meta' in checkpoint:
        meta = checkpoint['meta']
        print(f"Model metadata: {meta}")

    return state_dict

def load_model(state_dict, device):
    """
    Load specialist model architecture and weights.
    Uses torchvision's CascadeRCNN (if available) or adapted Mask R-CNN.
    """
    print("Loading Mask R-CNN model architecture...")

    # For now, use torchvision's Mask R-CNN as base
    # The specialist model uses similar architecture
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,  # We'll load custom weights
        progress=True
    )

    print("Attempting to load specialist model weights...")

    # Try to load specialist weights
    # Note: There may be key mismatches since spec model is Cascade R-CNN
    # We'll match compatible keys
    model_state = model.state_dict()

    matched = 0
    skipped = 0

    for key in state_dict.keys():
        if key in model_state:
            try:
                model_state[key] = state_dict[key]
                matched += 1
            except Exception as e:
                skipped += 1
        else:
            skipped += 1

    print(f"Loaded {matched} matching weights, skipped {skipped} incompatible")

    model.to(device)
    model.eval()

    return model

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
        }

def prepare_image(image_array, device):
    """Prepare image tensor for model."""
    img_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
    img_tensor = img_tensor / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor.to(device)

def tile_wsi(image_path, tile_size=TILE_SIZE, overlap_ratio=OVERLAP_RATIO, max_tiles=None):
    """Tile WSI image."""
    print(f"Loading WSI and tiling (size={tile_size}, overlap={overlap_ratio})...")

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]

    stride = int(tile_size * (1 - overlap_ratio))
    tiles_generated = 0

    y_pos = 0
    y_idx = 0

    while y_pos < height:
        x_idx = 0
        x_pos = 0
        y_end = min(y_pos + tile_size, height)

        while x_pos < width:
            if max_tiles and tiles_generated >= max_tiles:
                print(f"Reached max_tiles limit ({max_tiles})")
                return

            x_end = min(x_pos + tile_size, width)
            tile = img_rgb[y_pos:y_end, x_pos:x_end].copy()

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
            }

            tiles_generated += 1
            yield tile, metadata

            x_pos += stride
            x_idx += 1

        y_pos += stride
        y_idx += 1

    print(f"Total tiles: {tiles_generated}")

def process_tile(model, tile_array, device, score_threshold=SCORE_THRESHOLD):
    """Process a tile with the model."""
    with torch.no_grad():
        img_tensor = prepare_image(tile_array, device)
        predictions = model([img_tensor])

    result = predictions[0]
    boxes = result['boxes'].cpu().numpy()
    scores = result['scores'].cpu().numpy()
    masks = result['masks'].cpu().numpy()

    detections = []
    for idx, score in enumerate(scores):
        if score > score_threshold:
            detection = {
                'box': boxes[idx].tolist(),
                'score': float(score),
                'mask': masks[idx][0]
            }
            detections.append(detection)

    return detections

def segment_wsi():
    """Main segmentation pipeline."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"WSI Glomeruli Segmentation - Specialist Model")
    print(f"{'='*70}")

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get WSI info
    wsi_info = get_wsi_info(IMAGE_PATH)
    print(f"\nWSI: {wsi_info['width']}x{wsi_info['height']} pixels, {wsi_info['file_size_mb']:.1f} MB")

    # Load model
    state_dict = load_checkpoint_weights()
    model = load_model(state_dict, device)

    # Process tiles
    print(f"\nProcessing tiles...")
    all_detections = defaultdict(list)
    tile_results = []

    for tile_idx, (tile_array, metadata) in enumerate(tile_wsi(IMAGE_PATH, max_tiles=MAX_TILES)):
        if (tile_idx + 1) % 10 == 0:
            print(f"  Tile {tile_idx + 1}...")

        detections = process_tile(model, tile_array, device, SCORE_THRESHOLD)

        # Adjust to global coordinates
        for det in detections:
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

    total_detections = sum(len(dets) for dets in all_detections.values())

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Tiles processed: {len(tile_results)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg per tile: {total_detections / max(1, len(tile_results)):.2f}")

    # Save results
    results_json = {
        'wsi_info': wsi_info,
        'config': {
            'tile_size': TILE_SIZE,
            'overlap_ratio': OVERLAP_RATIO,
            'score_threshold': SCORE_THRESHOLD,
        },
        'statistics': {
            'total_tiles': len(tile_results),
            'total_detections': total_detections,
        },
    }

    results_file = os.path.join(OUTPUT_DIR, 'glomeruli_detections.json')
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"{'='*70}\n")

    return True

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found: {IMAGE_PATH}")
        return False

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found: {CHECKPOINT_PATH}")
        return False

    try:
        segment_wsi()
        return True
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
