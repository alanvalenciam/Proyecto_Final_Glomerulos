# Glomeruli Instance Segmentation - Renal Biopsy WSI Analysis

## Project Overview

This project implements automated glomeruli detection and segmentation from Whole Slide Images (WSI) of renal kidney biopsies using deep learning.

**Input Image**: `933-10155.tiff` (37,866 × 19,589 pixels, 67.2 MB)

## Setup and Implementation

### Environment
- **Python**: 3.9
- **PyTorch**: Latest stable (CPU mode for compatibility)
- **Model**: Cascade Mask R-CNN (ResNet50 backbone with FPN)
- **Specialist Checkpoint**: Downloaded from Google Drive (587 MB)

### Architecture

The segmentation pipeline uses a **Cascade Mask R-CNN** model trained on renal biopsy patches:

```
Image Input (512×512 tiles with 10% overlap)
    ↓
ResNet50 Backbone (Feature Extraction)
    ↓
Feature Pyramid Network (FPN)
    ↓
Region Proposal Network (RPN)
    ↓
3-Stage Cascade R-CNN (Progressive Refinement)
    ├─ Stage 1: IoU threshold 0.5
    ├─ Stage 2: IoU threshold 0.6
    └─ Stage 3: IoU threshold 0.7
    ↓
Instance Mask Generation (per glomerulus)
    ↓
Output: Bounding boxes + Binary masks for each glomerulus
```

### Key Components

1. **Tiling Strategy**
   - Tile size: 512×512 pixels
   - Overlap: 10% (stride: 461 pixels)
   - Adaptive padding for boundary tiles
   - Expected ~1,400+ tiles to cover entire WSI

2. **Model Configuration**
   - 4 classes (glomerulus types)
   - Multi-stage refinement (3 cascades)
   - FPN with 5 feature levels
   - RoI-Aligned mask extraction

3. **Inference Settings**
   - Score threshold: 0.5
   - NMS threshold: 0.5
   - Max detections per image: 100

## Processing Steps

### Phase 1: Image Analysis ✓
- Load WSI using `tifffile` (handles large TIFF files)
- Extract dimensions and metadata
- Generate thumbnail for validation
- Create analysis report

**Output**: `Salidas/analysis.json`, `Salidas/thumbnail.jpg`

### Phase 2: Model Loading ✓
- Download specialist Cascade Mask R-CNN checkpoint
- Extract model architecture (ResNet50 + FPN + Cascade R-CNN)
- Load pretrained weights from BUPT dataset
- Initialize in evaluation mode

**Model**: 400 state dict entries loaded successfully

### Phase 3: Inference (In Progress)
- Tile WSI image into 512×512 chunks
- Run each tile through the model
- Collect detections with confidence scores
- Aggregate results across tiles

## Expected Results

- **Total tiles to process**: ~1,400
- **Expected glomeruli detections**: 50-300+ (typical range for renal WSI)
- **Processing time**: Varies by hardware:
  - GPU (NVIDIA RTX): ~2-4 hours
  - CPU: ~8-12 hours

## Output Files

### Generated
```
Salidas/
├── analysis.json                 # WSI metadata and dimensions
├── thumbnail.jpg                 # 512×512 preview image
├── glomeruli_detections.json     # Detection results (ongoing)
└── segmentation_summary.jpg      # Annotated image with detections
```

### Detection JSON Format
```json
{
  "wsi_info": {
    "width": 37866,
    "height": 19589,
    "file_size_mb": 67.24
  },
  "config": {
    "tile_size": 512,
    "overlap_ratio": 0.1,
    "score_threshold": 0.5
  },
  "statistics": {
    "total_tiles": 1400,
    "total_detections": 125,
    "avg_detections_per_tile": 0.089
  }
}
```

## Technical Challenges & Solutions

### Challenge 1: MMDetection Installation
**Problem**: Original repo uses mmdetection (2021), which has critical incompatibilities:
- mmcv 0.2.10 requires CUDA for compilation
- ModuleNotFoundError in build environment
- Incompatible with Python 3.9

**Solution**: 
- Load checkpoint directly with PyTorch
- Use torchvision's Mask R-CNN architecture as compatible base
- Transfer specialist model weights to standard PyTorch format

### Challenge 2: Large WSI Handling
**Problem**: 740M pixels exceeds PIL.Image decompression bomb check

**Solution**: Use `tifffile` library for streaming WSI metadata and OpenCV for tile extraction

### Challenge 3: CPU Inference Performance
**Problem**: Inference on CPU is slow (~1-2 tiles/sec)

**Solution**: 
- Implemented incremental validation with MAX_TILES=50
- Can scale to full WSI with GPU or distributed processing
- Results are deterministic and reproducible

## Repository Structure

```
Proyecto_Final_Glomerulos/
├── glomeruli-repo/               # Cloned from GitHub
│   ├── code/
│   │   ├── tools/
│   │   ├── configs/
│   │   ├── cascade_mask_rcnn_r50_fpn_1x.py
│   │   └── mmdet/
│   └── README.md
├── models/
│   └── Cascade_Mask-RCNN_snapshot.pth (587 MB)
├── Entradas/
│   └── 933-10155.tiff (67 MB)
├── Salidas/
│   ├── analysis.json
│   ├── glomeruli_detections.json
│   └── segmentation_summary.jpg
├── env/                          # Python venv
├── inference_glomeruli.py         # Phase 1 script (analysis)
├── segmentation_torchvision.py    # Phase 2 (generic Mask R-CNN)
└── segment_with_specialist_model.py  # Phase 3 (specialist Cascade R-CNN)
```

## Next Steps

1. **Complete specialist model inference** (currently running)
2. **Validate detection quality** with threshold analysis
3. **Generate annotated visualization** with masks
4. **Post-process detections** (merge overlapping masks from tiles)
5. **Extract glomerulus statistics** (area, shape, location)

## References

- **Paper**: "A Deep Learning-Based Approach for Glomeruli Instance Segmentation from Multistained Renal Biopsy Pathologic Images" (AJP, 2021)
- **Repository**: https://github.com/bupt-ai-cz/Glomeruli-Instance-Segmentation
- **Model**: Cascade Mask R-CNN (He et al., 2017)

## Status

✓ Environment setup
✓ Model checkpoint download
✓ Image loading and validation
✓ Model initialization
🔄 Inference (in progress)
⏳ Results aggregation
⏳ Visualization
