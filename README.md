# U-Net Glomeruli Segmentation — Renal Biopsy WSI Analysis

Automated **semantic segmentation** of glomeruli (functional kidney units) from renal biopsy whole-slide images (WSI) using PyTorch U-Net with **dynamic memory-aware parallelism** for Windows Server deployment.

![Pipeline](https://img.shields.io/badge/Pipeline-5%20Steps-blue) ![Model](https://img.shields.io/badge/Model-U--Net-green) ![Loss](https://img.shields.io/badge/Loss-BCE%2BDice-orange) ![Metric](https://img.shields.io/badge/Metric-MeanIoU-brightgreen)

---

## 🚀 Quick Start (Windows Server)

```powershell
# 1. Prepare input data
mkdir Entradas
# Copy your WSI files (*.tiff) + annotations (*.geojson) to Entradas/

# 2. Run the full pipeline
python tiling_unet.py              # ~5-20 min
python augmentation.py             # ~10-30 min
python normalizacion.py            # ~10-15 min
python estandarizacion.py          # ~10-15 min

# 3. Train U-Net
jupyter notebook U-Net_Training.ipynb
# Open the notebook and run all cells (or uncomment the training cell for full 50-epoch training)
# Takes ~2-8 hours depending on batch size and hardware

# 4. Monitor training
tensorboard --logdir checkpoints
# Open http://localhost:6006 in browser
```

**Total time**: ~2-3 days end-to-end (including training)

For detailed walkthrough, see **[WORKFLOW.md](WORKFLOW.md)** ← Start here!

---

## 📋 What This Project Does

**Input**: Whole-slide images (TIFF) + physician annotations (GeoJSON)  
**Output**: Trained U-Net model that predicts glomerulus boundaries + class labels (0-4)  
**Metric**: Mean Intersection over Union (MeanIoU) per glomerulus class

```
WSI TIFF + GeoJSON
     ↓
[1] Tiling (1024×1024 tiles, 50% overlap)
     ↓
[2] Data Augmentation (6 deterministic + synthetic)
     ↓
[3] Color Normalization (Reinhard stain transfer)
     ↓
[4] Z-Score Standardization (per-channel, reversible)
     ↓
[5] U-Net Training (BCE + Dice loss, MeanIoU validation)
     ↓
Trained Model (best_model.pth) + Metrics Report
```

---

## 🏗️ Architecture: U-Net for Medical Image Segmentation

### Why U-Net?

Unlike **bounding-box detection** (YOLO, Mask R-CNN), U-Net performs **pixel-level semantic segmentation**:
- Delineates exact boundaries of glomerulus capillary tufts (Bowman capsule + glomerular tuft)
- Separates glomerulus from background and other kidney compartments
- Handles massive class imbalance (background >> glomerulus pixels)

### Model Details

```
Input:  [B, 3, 1024, 1024]  (Z-score normalized RGB tiles)
         ↓
Encoder (4 levels):
  - DoubleConv blocks (Conv→BN→ReLU×2)
  - MaxPool2d for downsampling
         ↓
Bottleneck: (highest compression, full context)
         ↓
Decoder (4 levels):
  - Bilinear upsample
  - Skip connections (from encoder)
  - DoubleConv blocks
         ↓
Output: [B, 5, 1024, 1024]  (logits: background + 4 glomerulus classes)
```

### Loss Function: Combined BCE + Dice

```python
Loss = 0.5 * CrossEntropyLoss + 0.5 * DiceLoss

CrossEntropyLoss:
  - Strict pixel classification (per-class probability)
  - Class weights: inverse frequency (handles imbalance)

DiceLoss:
  - Dice coefficient = 2|X∩Y| / (|X|+|Y|)  (IoU proxy)
  - Penalizes lack of geometric overlap
  - Critical for detecting small glomeruli amid background
```

### Evaluation: Mean Intersection over Union (MeanIoU)

Per-class IoU: `IoU_c = intersection_c / (union_c)` for each class c ∈ {0,1,2,3,4}

**MeanIoU** = average IoU across all classes

- **Background (class 0)**: large but lower priority
- **Glomerulus classes (1-4)**: small but clinically critical → balanced by class weighting

---

## 📁 Project Structure

```
Proyecto_Final_Glomerulos/
├── Entradas/                          ← Input: WSI TIFF + GeoJSON
│   ├── slide_001.tiff
│   ├── slide_001.geojson
│   └── ...
│
├── Salidas/                           ← All outputs
│   ├── Tiles_UNet/                    ← Step 1: Tiled + labeled
│   ├── dataset_aug/                   ← Step 2: Augmented
│   ├── Normalizados/                  ← Step 3: Color-normalized
│   ├── Estandarizados/                ← Step 4: Z-score standardized
│   └── checkpoints/                   ← Step 5: Trained models
│
├── tiling_unet.py                     ← Extract tiles + masks from WSI
├── augmentation.py                    ← Data augmentation (6+3 transforms)
├── normalizacion.py                   ← Reinhard color normalization
├── estandarizacion.py                 ← Z-score standardization
│
├── U-Net_Training.ipynb               ← Complete training notebook
│                                         (includes dataset, losses, U-Net model, training loop)
│
├── README.md                          ← This file
├── WORKFLOW.md                        ← Detailed Windows Server guide
```

---

## 💾 System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Python | 3.10+ | 3.11 (Windows) |
| RAM | 16 GB | 32+ GB |
| Disk | 500 GB | 1 TB+ |
| CPU cores | 8 | 16+ |
| GPU (optional) | — | NVIDIA 4GB VRAM |

### Python Packages

```bash
pip install torch torchvision opencv-python numpy scikit-image scikit-learn
pip install albumentations shapely tensorboard psutil tqdm
```

---

## 🔄 Dynamic Parallelism (Why It Matters)

Every processing script uses **memory-aware dynamic parallelism**:

```
┌─ Compute RAM budget (once at startup)
│
├─ For each task:
│  ├─ Check if RAM allows next task (dual check: software counter + live OS)
│  ├─ If yes → submit task
│  └─ If no → WAIT (don't crash with OOM)
│
└─ When task finishes → decrement RAM, try next task
```

**Why?** On Windows servers with shared resources:
- ✅ Prevents out-of-memory crashes
- ✅ Detects external memory pressure (other processes)
- ✅ Keeps safety margin (1GB free) to prevent OS swapping
- ✅ Clear logging: `"Tile 18/847 finished. Freeing 25.2MB. Trying to admit tile 19..."`

**Tune with**: `--ram-fraction 0.75` and `--min-free-gb 1.0`

---

## 📊 Training Pipeline

### Data Preparation
1. **Tiling**: 1024×1024 tiles from WSI with 50% overlap
2. **Filtering**: Skip background-only tiles (Otsu thresholding on tissue mask)
3. **Augmentation**:
   - **Component A** (deterministic): 6 transforms (flip, rotate) on ALL tiles
   - **Component B** (synthetic): 0-3 random augmentations if class is underrepresented
4. **Normalization**: Reinhard stain color transfer (tissue color consistency)
5. **Standardization**: Z-score per RGB channel (reversible uint8 encoding)

### Data Split
- **Train**: 70% biopsies (not tiles!) 
- **Val**: 15% biopsies
- **Test**: 15% biopsies

Why by biopsia? Tiles from same slide are correlated → splits by biopsia prevent leakage.

### Training Loop
- **Optimizer**: AdamW (weight_decay=1e-4)
- **LR Schedule**: Linear warmup (5 epochs) → cosine annealing
- **Loss**: 50/50 BCE + Dice
- **Class Weights**: Inverse frequency (handles background/glomerulus imbalance)
- **Validation**: MeanIoU per epoch, checkpoint best model
- **Logging**: TensorBoard (loss, MeanIoU, per-class IoU)

---

## 🖥️ Windows Server Notes

### Path Handling
All scripts use `Path()` (pathlib) which handles `/` and `\` correctly:
```python
from pathlib import Path
Path("Salidas") / "Tiles_UNet"  # Works on Windows and Linux
```

### OpenSlide on Windows
Download from: https://openslide.cs.cmu.edu/download/openslide-winbuild/

Add to PATH:
```powershell
$env:PATH += ";C:\openslide-winbuild-20230414\bin"
```

### CUDA Detection
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If not detected, reinstall PyTorch matching your CUDA version.

### Disk I/O Performance
- Use **SSD** (not HDD) for `Salidas/` → tiles are written many times
- Avoid network storage (NAS/SMB) → local disk is 10x faster
- Monitor: `python -c "import shutil; print(shutil.disk_usage('.'))"` (GB free)

---

## 📖 Usage Examples

### Run Full Pipeline with Defaults
```bash
python tiling_unet.py
python augmentation.py
python normalizacion.py
python estandarizacion.py
jupyter notebook U-Net_Training.ipynb
# Open the notebook and run all cells to train the U-Net
```

### Custom Configuration
```bash
# Fewer workers if RAM is tight
python normalizacion.py --workers 4 --ram-fraction 0.5

# Run only on specific biopsy
python estandarizacion.py --input Salidas/Normalizados/biopsia_001

# For training configuration (epochs, batch size, learning rate):
# Open U-Net_Training.ipynb and modify the parameters in the training cell
```

### Resume Training
```bash
# Open U-Net_Training.ipynb and modify the notebook code to load a checkpoint
# before training (set the resume path in the training cell)
```

---

## 🔍 Outputs

### After Step 1 (Tiling)
```
Salidas/Tiles_UNet/biopsia_001/
├── images/
│   ├── slide_001_tile_0_0.png   (1024×1024 RGB)
│   └── ...
├── masks/
│   ├── slide_001_tile_0_0_mask.png  (class IDs 0-4)
│   └── ...
└── annotations.json             (tile registry, glomeruli mapping)
```

### After Step 5 (Training)
```
checkpoints/unet_YYYYMMDD_HHMMSS/
├── unet_YYYYMMDD_HHMMSS_best.pth      (model weights)
├── unet_YYYYMMDD_HHMMSS_last.pth      (last checkpoint)
├── unet_YYYYMMDD_HHMMSS_report.json   (final metrics)
└── runs/                               (TensorBoard logs)
    ├── events.out.tfevents.1714...
    └── ...
```

**Metrics file** (JSON):
```json
{
  "final_metrics": {
    "train_loss": 0.234,
    "val_loss": 0.312,
    "val_mean_iou": 0.687,
    "test_mean_iou": 0.672,
    "test_per_class_iou": {
      "class_0": 0.95,
      "class_1": 0.62,
      "class_2": 0.58,
      "class_3": 0.54,
      "class_4": 0.51
    }
  }
}
```

---

## 🐛 Troubleshooting

### "MemoryError" or "Process killed"
→ Reduce `--ram-fraction` to 0.5, or increase `--min-free-gb` to 2.0  
→ Check if other processes are consuming RAM: `tasklist /V | findstr python`

### "No PNG images found"
→ Verify directory structure: `dir Salidas\dataset_aug\biopsia_001\images\`  
→ Check that previous step completed successfully

### "OpenSlide: library not found"
→ Download and install OpenSlide, add to PATH  
→ Verify: `python -c "import openslide; print(openslide.__file__)"`

### GPU not detected
→ `python -c "import torch; print(torch.cuda.is_available())"`  
→ Reinstall PyTorch matching your CUDA version

For more troubleshooting, see **[WORKFLOW.md](WORKFLOW.md)**.

---

## 📚 References

### Architecture
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
- **Dice Loss**: Sørensen–Dice coefficient for segmentation
- **Medical Image Analysis**: Standard practice for pathology image segmentation

### Medical Context
- **Glomeruli**: Functional units of kidney; assessment critical for diagnosis
- **Bowman's Capsule**: Surrounds glomerular tuft; key landmark for segmentation
- **Biopsy Classification**: 4 classes based on pathological features

### Related Work
- Instance segmentation: Mask R-CNN
- Semantic segmentation: DeepLab, PSPNet
- Medical imaging: nnU-Net (state-of-the-art for medical tasks)

---

## 📝 Documentation

- **[WORKFLOW.md](WORKFLOW.md)** — Complete Windows Server deployment guide (full pipeline walkthrough, CLI reference, troubleshooting)
- **[README.md](README.md)** — This file (architecture, quick start, system requirements)

---

## 🎯 Next Steps

1. **Prepare data**: Place WSI + GeoJSON pairs in `Entradas/`
2. **Run pipeline**: Follow Quick Start section above
3. **Monitor training**: Open TensorBoard while training
4. **Evaluate results**: Check `checkpoints/*_report.json` for final metrics
5. **Deploy model**: Use `best_model.pth` for inference on new WSI

---

## 📄 License & Attribution

This project implements semantic segmentation for renal biopsy analysis using U-Net with PyTorch.

**Key innovations**:
- Dynamic memory-aware parallelism for Windows Server
- Combined BCE+Dice loss for class imbalance
- Per-biopsia data split (prevents leakage)
- Reversible Z-score encoding

---

**Questions?** Check [WORKFLOW.md](WORKFLOW.md) for detailed instructions.  
**Issues?** See Troubleshooting section above.

---

*Last updated: 2026-05-07*  
*Python 3.10+ | PyTorch 2.0+ | Windows/Linux/macOS*
