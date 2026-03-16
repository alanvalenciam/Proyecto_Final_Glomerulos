# -*- coding: utf-8 -*-
"""
WSI → Recortes centrados por glomérulo
Ejecutar DESPUÉS de exportar las anotaciones desde QuPath.

Requisitos:
    pip install openslide-python pillow numpy

    En Windows también necesitas instalar OpenSlide binaries:
    https://openslide.org/api/python/#installing
"""

from pathlib import Path
from typing import List
import numpy as np
import openslide
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# ──────────────────────────────────────────────
# Carga de anotaciones YOLO
# ──────────────────────────────────────────────

def load_yolo_annotations(wsi_path: str):
    annot_path = Path(wsi_path).with_suffix('.txt')
    if not annot_path.exists():
        raise FileNotFoundError(
            f"\n❌ No se encontró el archivo de anotaciones: {annot_path}"
            f"\n   Asegúrate de que el .txt esté en la misma carpeta que el .tiff"
            f"\n   y tenga el mismo nombre: BR-007-HYE-25-CONV.txt"
        )

    with openslide.OpenSlide(wsi_path) as slide:
        W, H = slide.dimensions

    anns = []
    with open(annot_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            anns.append([
                (cx - w / 2) * W,
                (cy - h / 2) * H,
                (cx + w / 2) * W,
                (cy + h / 2) * H,
                int(cls)
            ])

    print(f"  → {len(anns)} glomérulos cargados desde {annot_path.name}")
    return anns, W, H


# ──────────────────────────────────────────────
# Filtro de tejido
# ──────────────────────────────────────────────

def _is_tissue(img_pil: Image.Image, min_tissue: float = 0.20) -> bool:
    hsv = np.array(img_pil.convert("HSV"))
    S = hsv[..., 1]
    V = hsv[..., 2]
    tissue_mask = (S > 20) & (V > 25) & (V < 235)
    return float(tissue_mask.sum()) / tissue_mask.size > min_tissue


# ──────────────────────────────────────────────
# Extractor principal
# ──────────────────────────────────────────────

def extract_glom_crops(
    wsi_path: str,
    output_dir: str,
    target_downsample: float = 2.0,
    crop_size: int = 512,
    padding_factor: float = 1.6,
    min_tissue_ratio: float = 0.20,
    save_labels: bool = True,
):
    out     = Path(output_dir)
    img_dir = out / "images";  img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir = out / "labels";  lbl_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(wsi_path).stem
    anns, W0, H0 = load_yolo_annotations(wsi_path)

    saved   = 0
    skipped = 0

    NOMBRES_CLASE = {
        0: "Normal",
        1: "Global_Sclerosis",
        2: "Segmental_Sclerosis"
    }

    with openslide.OpenSlide(wsi_path) as slide:
        level = slide.get_best_level_for_downsample(target_downsample)
        dL    = float(slide.level_downsamples[level])

        for idx, (x1, y1, x2, y2, cid) in enumerate(anns):
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                skipped += 1
                continue

            # Centro y tamaño del recorte en nivel 0
            cx0   = (x1 + x2) / 2
            cy0   = (y1 + y2) / 2
            side0 = max(bw, bh) * padding_factor
            half0 = side0 / 2

            rx0    = int(max(0, cx0 - half0))
            ry0    = int(max(0, cy0 - half0))
            rx0    = min(rx0, int(W0 - side0))
            ry0    = min(ry0, int(H0 - side0))
            rsize0 = int(side0)

            # Tamaño en el nivel elegido
            rw_level = max(1, int(round(rsize0 / dL)))

            try:
                region = slide.read_region(
                    (rx0, ry0), level, (rw_level, rw_level)
                ).convert("RGB")
            except Exception as e:
                print(f"  ⚠️  Error en glomérulo #{idx}: {e}")
                skipped += 1
                continue

            crop = region.resize((crop_size, crop_size), Image.LANCZOS)

            if not _is_tissue(crop, min_tissue_ratio):
                skipped += 1
                continue

            # Guardar imagen
            name = f"{stem}_glom_{idx:04d}_cls{cid}"
            crop.save(img_dir / f"{name}.png", "PNG", optimize=True, compress_level=6)

            # Guardar label YOLO re-normalizado
            if save_labels:
                lines = []
                for (ax1, ay1, ax2, ay2, acid) in anns:
                    ox1 = max(ax1, rx0);       oy1 = max(ay1, ry0)
                    ox2 = min(ax2, rx0+rsize0); oy2 = min(ay2, ry0+rsize0)
                    if ox2 <= ox1 or oy2 <= oy1:
                        continue
                    vis = ((ox2-ox1)*(oy2-oy1)) / ((ax2-ax1)*(ay2-ay1) + 1e-9)
                    if vis < 0.25:
                        continue
                    rcx = ((ox1+ox2)/2 - rx0) / rsize0
                    rcy = ((oy1+oy2)/2 - ry0) / rsize0
                    rw_ = (ox2-ox1) / rsize0
                    rh_ = (oy2-oy1) / rsize0
                    lines.append(
                        f"{acid} "
                        f"{min(max(rcx,0),1):.6f} "
                        f"{min(max(rcy,0),1):.6f} "
                        f"{min(max(rw_,1e-5),1):.6f} "
                        f"{min(max(rh_,1e-5),1):.6f}"
                    )
                (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

            clase_nombre = NOMBRES_CLASE.get(cid, f"clase_{cid}")
            print(f"  [{saved+1:03d}] Glomérulo #{idx} → {clase_nombre} → {name}.png")
            saved += 1

    print()
    print("=" * 55)
    print(f"✅  Recortes guardados  : {saved}")
    print(f"⚠️   Descartados        : {skipped}")
    print(f"📁  Imágenes en        : {img_dir}")
    print(f"📁  Labels en          : {lbl_dir}")
    print("=" * 55)


# ──────────────────────────────────────────────
# EJECUTAR
# ──────────────────────────────────────────────

if __name__ == "__main__":

    WSI_PATH   = r"D:\Anotaciones\BR-007-HYE-25-CONV.tiff"
    OUTPUT_DIR = r"D:\Anotaciones\glom_crops"

    extract_glom_crops(
        wsi_path          = WSI_PATH,
        output_dir        = OUTPUT_DIR,
        target_downsample = 2.0,    # 1.0 = full-res (lento), 2.0 = recomendado
        crop_size         = 512,    # tamaño final de cada recorte en píxeles
        padding_factor    = 1.6,    # margen alrededor del glomérulo (1.0 = sin margen)
        min_tissue_ratio  = 0.20,   # descarta recortes con menos del 20% de tejido
        save_labels       = True,   # guarda .txt YOLO por cada recorte
    )