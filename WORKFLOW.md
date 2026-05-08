# U-Net Glomeruli Segmentation — Workflow Documentation

Complete pipeline for segmenting glomeruli in whole-slide images (WSI) using PyTorch U-Net, with **dynamic memory-aware parallelism**.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
4. [Script Reference](#script-reference)
5. [Dynamic Parallelism](#dynamic-parallelism)
6. [Windows Server Notes](#windows-server-notes)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Python Environment
- **Python 3.10+** (3.11 recommended for Windows Server)
- **PyTorch 2.0+** with CUDA support (if GPU available)
- **OpenSlide** for WSI reading (see below)

### Required Packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy scikit-image scikit-learn albumentations shapely
pip install tensorboard psutil tqdm
```

### System Requirements (Windows Server)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 16 GB | 32+ GB |
| CPU cores | 8 | 16+ |
| GPU (optional) | VRAM 4 GB | VRAM 8+ GB |
| Storage | 500 GB | 1 TB+ (for tiles + checkpoints) |

### OpenSlide on Windows

Download from: https://openslide.cs.cmu.edu/download/openslide-winbuild/

Extract and add to PATH:
```powershell
# Example: C:\openslide-winbuild-20230414\bin
$env:PATH += ";C:\openslide-winbuild-20230414\bin"
```

Verify:
```bash
python -c "import openslide; print(openslide.__file__)"
```

---

## Project Structure

```
Proyecto_Final_Glomerulos/
├── Entradas/                          # Input: WSI TIFF + GeoJSON annotations
│   ├── slide_001.tiff
│   ├── slide_001.geojson
│   ├── slide_002.tiff
│   └── slide_002.geojson
│
├── Salidas/                           # All outputs
│   ├── Tiles_UNet/                    # Step 1: Tiled WSIs + masks
│   │   ├── biopsia_001/
│   │   │   ├── images/                # PNG tiles (1024×1024, RGB)
│   │   │   ├── masks/                 # PNG masks (1024×1024, class IDs 0-4)
│   │   │   └── annotations.json       # Metadata + tile registry
│   │   └── biopsia_002/
│   │
│   ├── Normalizados/                  # Step 2: Color-normalized tiles
│   │   ├── biopsia_001/
│   │   │   └── images/                # Reinhard color-normalized PNGs
│   │   └── biopsia_002/
│   │
│   ├── Estandarizados/                # Step 3: Z-score standardized tiles
│   │   ├── biopsia_001/
│   │   │   └── images/                # Z-score as uint8 (reversible encoding)
│   │   └── biopsia_002/
│   │
│   └── checkpoints/                   # Training outputs
│       └── unet_YYYYMMDD_HHMMSS/
│           ├── unet_YYYYMMDD_HHMMSS_best.pth
│           ├── unet_YYYYMMDD_HHMMSS_last.pth
│           ├── unet_YYYYMMDD_HHMMSS_report.json
│           └── runs/                  # TensorBoard logs
│
├── tiling_unet.py                     # Step 1: WSI → tiles
├── normalizacion.py                   # Step 2: Color normalization
├── estandarizacion.py                 # Step 3: Z-score standardization
│
├── U-Net_Training.ipynb               # Step 4: Complete training notebook
│                                         (includes dataset, losses, U-Net model, training loop)
│
└── WORKFLOW.md                        # This file
```

---

## Full Pipeline Walkthrough

### Step 0: Prepare Input Data

Create `Entradas/` directory with matched pairs of TIFF + GeoJSON:

```powershell
cd C:\path\to\Proyecto_Final_Glomerulos
mkdir Entradas
mkdir Salidas

# Copy your WSI files (*.tiff) and annotations (*.geojson) to Entradas/
# Files must have matching names: slide_001.tiff + slide_001.geojson
```

### Step 1: Tiling (WSI → 1024×1024 tiles)

```bash
python tiling_unet.py --input Entradas/ --output Salidas/Tiles_UNet/
```

**What it does:**
- Loads each WSI TIFF file
- Creates a sliding-window grid of 1024×1024 tiles (stride=512, 50% overlap)
- Rasterizes GeoJSON polygons into binary masks
- Filters out background-only tiles (Otsu thresholding)
- Produces: `Salidas/Tiles_UNet/{biopsia}/images/` + `masks/` + `annotations.json`

**Parallelism:** Dynamic (ProcessPoolExecutor + RAM backpressure). Watches memory after each tile and waits if RAM is low.

**Time:** ~5–20 minutes per slide (depending on slide size, CPU, disk speed)

---

### Step 2: Color Normalization (Reinhard)

```bash
python normalizacion.py --input Salidas/Tiles_UNet --output Salidas/Normalizados
```

**What it does:**
- Corrects staining variability across biopsies
- Computes LAB color statistics from a template (auto-selected 200 tiles or from `referencias/` folder)
- Applies Reinhard color transfer to each tile

**Parallelism:** Dynamic (ProcessPoolExecutor + RAM backpressure). Processes tiles in parallel, checking memory after each one.

**Time:** ~10–15 minutes

**Note:** Masks are NOT processed here (binary labels don't need color normalization). They stay in `Salidas/Tiles_UNet/` and are loaded directly during training.

---

### Step 3: Z-Score Standardization

```bash
python estandarizacion.py --input Salidas/Normalizados --output Salidas/Estandarizados
```

**What it does:**
- Computes per-channel RGB statistics from a random sample of tiles (tissue pixels only)
- Applies Z-score: `(pixel - mean) / std` per channel
- Encodes as uint8: `uint8 = (float32 + 4) / 8 * 255` (reversible: `float32 = (uint8 / 255 * 8) - 4`)

**Parallelism:** Dynamic (ProcessPoolExecutor + RAM backpressure).

**Time:** ~10–15 minutes

---

### Step 4: Training (U-Net)

```bash
jupyter notebook U-Net_Training.ipynb
```

**What it does (in the notebook):**
- Loads standardized images + masks, split 70% train / 15% val / 15% test (stratified by biopsia)
- Applies online augmentation (flips, rotations, noise) to train split only — never to val or test
- Defines U-Net architecture (4 encoder + 4 decoder levels, 2 output classes)
- Implements combined BCE + Dice loss function with per-class weights for imbalance
- Trains with AdamW + cosine annealing LR schedule + 5-epoch linear warmup
- Evaluates with Binary Segmentation Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Checkpoints best model (by val F1) + last model

**To run training:**
1. Open the notebook with `jupyter notebook U-Net_Training.ipynb`
2. Modify parameters in the training cell if needed (epochs, batch size, learning rate)
3. Run all cells from top to bottom, or uncomment the full training cell for 50-epoch training
4. The notebook will generate progress output and save checkpoints

**Output:**
- `checkpoints/unet_YYYYMMDD_HHMMSS_best.pth` — best validation model
- `checkpoints/unet_YYYYMMDD_HHMMSS_report.json` — final metrics
- TensorBoard logs in `runs/` subdirectory

**Parallelism:** DataLoader uses dynamic `num_workers` (calculated from available RAM).

**Time:** ~2–8 hours (depends on epochs, dataset size, GPU)

**Monitor training:**
```bash
# In a separate terminal:
tensorboard --logdir checkpoints
# Open http://localhost:6006 in browser
```

---

## Script Reference

### Pre-processing Scripts

#### `tiling_unet.py`

```bash
python tiling_unet.py [OPTIONS]

Options:
  --input TEXT                Path to input directory (Entradas/)
  --output TEXT               Path to output directory (Salidas/Tiles_UNet/)
  --workers INT               Number of parallel workers (auto-calculated)
  --ram-fraction FLOAT        Fraction of RAM to use (0.75 default)
  --tile-size INT             Tile size in pixels (1024 default)
  --stride INT                Sliding window stride (512 default, 50% overlap)
  --zoom FLOAT                Zoom scale for OpenSlide (0.5 default)
  --help                      Show this help
```

---

#### `normalizacion.py`

```bash
python normalizacion.py [OPTIONS]

Options:
  --input PATH                Path to input tiles (default: Salidas/Tiles_UNet)
  --output PATH               Path to output (default: Salidas/Normalizados)
  --template PATH             Single template image (optional)
  --workers INT               Parallel workers (auto-calculated)
  --help                      Show this help
```

---

#### `estandarizacion.py`

```bash
python estandarizacion.py [OPTIONS]

Options:
  --input PATH                Path to input tiles (default: Salidas/Normalizados)
  --output PATH               Path to output (default: Salidas/Estandarizados)
  --workers INT               Parallel workers (auto-calculated)
  --help                      Show this help
```

---

### Training

#### `U-Net_Training.ipynb`

Complete Jupyter notebook containing:
- **Dataset loading**: PyTorch Dataset loader with stratified train/val/test split by biopsia
- **Online augmentation**: Applied to train split only (HorizontalFlip, VerticalFlip, RandomRotate90, Transpose, Rotate, GaussNoise, GaussianBlur)
- **Loss functions**: Combined BCE + Dice loss with per-class weighting
- **U-Net architecture**: 4-level encoder-decoder with skip connections
- **Training loop**: AdamW optimizer, cosine annealing LR schedule, 5-epoch warmup
- **Evaluation**: Binary Segmentation Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Checkpointing**: Saves best and last models + metrics report

**To run:**
```bash
jupyter notebook U-Net_Training.ipynb
# Modify parameters in the training cell as needed
# Run all cells from top to bottom
```

**Key Parameters (modify in notebook cells):**
- `EPOCHS`: Number of training epochs (default: 50)
- `BATCH_SIZE`: Batch size (default: 4)
- `LEARNING_RATE`: Initial learning rate (default: 1e-3)
- `IMAGES_DIR`: Path to standardized images (default: Salidas/Estandarizados)
- `MASKS_DIR`: Path to masks (default: Salidas/Tiles_UNet)
- `OUTPUT_DIR`: Checkpoint output directory (default: checkpoints)

**Output:**
- `checkpoints/unet_YYYYMMDD_HHMMSS_best.pth` — best validation model
- `checkpoints/unet_YYYYMMDD_HHMMSS_report.json` — final metrics (loss, per-class metrics)
- TensorBoard logs in `runs/` subdirectory for visualization

---

## Dynamic Parallelism

Every processing script uses **memory-aware dynamic parallelism**:

1. **At startup:** Compute available RAM, size worker pool accordingly
2. **Before each task:** Check if RAM budget allows the task
3. **If RAM full:** Wait for running tasks to finish, then retry
4. **When task finishes:** Decrement RAM counter, immediately try next task

This prevents out-of-memory crashes on Windows servers with many processes.

### What to expect

When you run `python normalizacion.py`, you'll see:

```
[INFO] RAM-aware: 31.45GB available, ~25.2MB per worker → 18 workers
[INFO] Admission queue: 847 tiles. RAM budget: 23.59 GB
[INFO] Submitted 18 tiles (estimated RAM: 0.46 GB used)
[INFO] Tile 18/847 finished. Freeing 25.2MB. Trying to admit tile 19...
[INFO] Tile 19 fits in budget. Submitting...
```

The script throttles task submission to stay under the RAM limit.

---

## Windows Server Notes

### 1. Path Separators

All scripts use `Path()` (from `pathlib`) which handles `/` and `\` correctly:

```python
# Both work correctly on Windows:
Path("Salidas") / "Tiles_UNet" / "biopsia_001"
Path("Salidas\\Tiles_UNet\\biopsia_001")
```

Always use forward slashes `/` in command-line arguments — they work everywhere.

### 2. Long Filenames

Windows has a 260-character path limit (MAX_PATH). Glomeruli annotations with long polygon names might exceed this. Use `\\?\` prefix to bypass:

```python
# Internal to Python (handled automatically):
import os
os.environ['PYTHONLEGACYWINDOWSSTDIOENCODING'] = '1'
```

### 3. CUDA on Windows

If your server has a GPU, PyTorch should detect it automatically:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

If CUDA is not detected:
- Verify NVIDIA driver version matches PyTorch requirement
- Reinstall PyTorch with correct CUDA version: `pip install torch ... --index-url https://download.pytorch.org/whl/cu118`

### 4. Disk I/O Performance

Tiles are written to disk many times. On Windows servers:
- **SSD recommended** for input/output directories (much faster than HDD)
- Consider using a local SSD, not network storage (NAS/SMB), for `Salidas/` to avoid network latency
- Monitor disk usage: `python -c "import shutil; print(shutil.disk_usage('.'))"` — total dataset ~500GB–2TB depending on WSI count

### 5. Process Priority (Optional)

On shared Windows servers, lower the priority of long-running jobs to avoid blocking other users:

```powershell
# PowerShell: start Jupyter with BELOW_NORMAL priority
$p = Start-Process python -ArgumentList "-m jupyter notebook U-Net_Training.ipynb" -PassThru
(Get-Process -Id $p.Id).ProcessorAffinity = 0  # Optional: pin to specific cores
wmic process where processid=$($p.Id) call setpriority 1  # BELOW_NORMAL
```

---

## Troubleshooting

### "MemoryError" or "Process killed"

**Cause:** RAM budget was underestimated (tiles larger than expected, or other processes consuming memory).

**Fix:**
1. Reduce `--workers` manually: `python estandarizacion.py --workers 4` (instead of auto-calculated 18)
2. Increase `--ram-fraction` if you want more aggressive (e.g., 0.5 instead of 0.75)
3. Check system load: `tasklist /V | findstr python` — other Python processes eating RAM?
4. Increase available RAM or run on a less-loaded time

### "No PNG images found" or "Mask not found"

**Cause:** Path mismatch or missing intermediate output.

**Fix:**
1. Verify directory structure: `dir Salidas\Tiles_UNet\biopsia_001\images\`
2. Check that previous step completed successfully (look for `*.json` metadata files)
3. Verify relative paths: scripts should run from project root

### "OpenSlide: library not found"

**Cause:** OpenSlide not installed or not in PATH.

**Fix:**
1. Download OpenSlide from https://openslide.cs.cmu.edu/download/openslide-winbuild/
2. Extract to `C:\openslide-winbuild\`
3. Add to PATH: `$env:PATH += ";C:\openslide-winbuild\bin"`
4. Verify: `python -c "import openslide; print(openslide.__file__)"`

### Training is slow (GPU not used)

**Cause:** PyTorch not finding CUDA, or model running on CPU.

**Fix:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Open U-Net_Training.ipynb and set device='cuda' in the training cell
```

If CUDA still not available, reinstall PyTorch matching your CUDA version (check `nvidia-smi`).

### Windows Defender/antivirus slowing things down

**Cause:** Antivirus scanning files as they're written.

**Fix (careful!):**
Add `Salidas/` to Windows Defender exclusions:
```powershell
Add-MpPreference -ExclusionPath "C:\path\to\Salidas"
```

---

## Quick-Start Example

```bash
# From project root (C:\Proyecto_Final_Glomerulos)

# 1. Verify setup
python -c "import torch; print(f'PyTorch OK, CUDA: {torch.cuda.is_available()}')"

# 2. Tiling (5–20 min)
python tiling_unet.py

# 3. Normalization (10–15 min)
python normalizacion.py

# 4. Standardization (10–15 min)
python estandarizacion.py

# 5. Training (2–8 hours, depending on epochs)
jupyter notebook U-Net_Training.ipynb
# Open the notebook, modify parameters if needed, and run all cells

# 6. Monitor in TensorBoard (in another terminal)
tensorboard --logdir checkpoints
# Open http://localhost:6006
```

Total end-to-end time: **~2–3 days** for a full dataset (tiling + preprocessing + 50 epochs training).

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Dice Loss for Medical Image Segmentation](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
- [OpenSlide](https://openslide.cs.cmu.edu/)

---

*Last updated: 2026-05-07*
*For questions or issues, check the project's issue tracker.*
