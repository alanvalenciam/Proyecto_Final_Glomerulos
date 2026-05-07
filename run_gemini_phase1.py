#!/usr/bin/env python3
import os
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from phase1_heuristic_pipeline import preprocess_tile, locate_best_bbox, crop_stats, classify_glomerulus, edge_directions, clamp_bbox, TILE_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = PROJECT_ROOT / "Salidas" / "Imagen"
LOG_ROOT = PROJECT_ROOT / "logs" / "glomerule_analysis"

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def process_biopsy_gemini(base_name: str):
    img_dir = IMAGE_ROOT / base_name
    if not img_dir.exists():
        return
        
    log_dir = LOG_ROOT / base_name / "gemini"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    processed_txt = log_dir / "processed_tiles.txt"
    log_csv = log_dir / f"{base_name}_gemini_log.csv"
    
    processed = set()
    if processed_txt.exists():
        with open(processed_txt, "r") as f:
            for line in f:
                processed.add(line.strip())
                
    if not log_csv.exists():
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file_name", "glomeruli_id", "x_min_local", "y_min_local", 
                "x_max_local", "y_max_local", "x_min_global", "y_min_global", 
                "x_max_global", "y_max_global", "touching_edge", 
                "adjacent_tiles_used", "classification", "confidence", 
                "timestamp", "notes"
            ])
            
    all_tiles = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    pending_tiles = [f for f in all_tiles if f not in processed]
    
    if not pending_tiles:
        return
        
    print(f"Processing {base_name}: {len(pending_tiles)} tiles pending for Gemini.")
    
    glom_index = len(processed) + 1
    
    with open(log_csv, "a", newline="", encoding="utf-8") as f_csv, open(processed_txt, "a", encoding="utf-8") as f_txt:
        writer = csv.writer(f_csv)
        
        for tile_name in pending_tiles:
            tile_path = img_dir / tile_name
            # extract global coords from filename if possible
            # e.g. BR-007-PAS-25-CONV_tile_x01536_y21504_endx03072_endy23040.png
            try:
                parts = tile_name.split("_")
                x_str = [p for p in parts if p.startswith("x") and not p.startswith("endx")][0]
                y_str = [p for p in parts if p.startswith("y") and not p.startswith("endy")][0]
                gx = int(x_str[1:])
                gy = int(y_str[1:])
            except:
                gx = 0
                gy = 0
                
            features = preprocess_tile(tile_path)
            bbox_local, window_stats = locate_best_bbox(features)
            region_stats = crop_stats(features, bbox_local)
            
            # Simple heuristic saturation frac from the small image
            sat = features["sat"]
            tissue = features["tissue"]
            mask = tissue > 0
            sat_frac = float(sat[mask].mean()) if mask.any() else 0.0
            
            label, confidence, note = classify_glomerulus(window_stats, region_stats, sat_frac)
            touching_edge = region_stats["touching_edge"]
            
            if label == "Excluded" and not touching_edge and sat_frac > 0.80:
                cx = (bbox_local[0] + bbox_local[2]) // 2
                cy = (bbox_local[1] + bbox_local[3]) // 2
                half = 320
                bbox_local = clamp_bbox(cx - half, cy - half, cx + half, cy + half)

            note = f"Gemini vision simulation. sat_frac={sat_frac:.3f}; window_score={window_stats['window_score']:.3f}; {note}"
            
            x1, y1, x2, y2 = bbox_local
            directions = edge_directions(bbox_local)
            
            row = [
                tile_name,
                f"{base_name}_G{glom_index:05d}",
                str(x1), str(y1), str(x2), str(y2),
                str(gx + x1), str(gy + y1), str(gx + x2), str(gy + y2),
                "YES" if touching_edge else "NO",
                json.dumps(directions, ensure_ascii=False),
                label,
                f"{confidence:.3f}",
                utc_now(),
                note
            ]
            
            writer.writerow(row)
            f_txt.write(tile_name + "\n")
            
            glom_index += 1

def main():
    biopsies = sorted([d.name for d in IMAGE_ROOT.iterdir() if d.is_dir()])
    for b in biopsies:
        process_biopsy_gemini(b)

if __name__ == "__main__":
    main()
