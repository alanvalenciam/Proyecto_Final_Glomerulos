#!/usr/bin/env python3
"""
Fase 1 detallada (heurística) para análisis de glomérulos.

Contexto del proyecto:
- El barrido inicial ya generó candidate tiles y progreso por tile.
- Los logs glomerulares detallados siguen vacíos.
- No hay pesos entrenados ni dependencias CV pesadas en el repo.

Este script completa una versión reproducible de la Fase 1 usando:
1. NMS espacial sobre `*_candidate_tiles.csv` para escoger tiles representantes.
2. Localización aproximada de un ROI glomerular dentro del tile.
3. Clasificación heurística a 4 clases:
   - Non-Proliferative  (ISN/RPS I, II, V)
   - Proliferative      (ISN/RPS III, IV)
   - Sclerosed          (ISN/RPS VI)
   - Excluded           (artefacto/tangencial/incompleto)

Importante:
- Es un pipeline heurístico de preanotación, NO un sustituto de validación experta.
- Está pensado para desbloquear la trazabilidad de Fase 1 y dejar un log auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = PROJECT_ROOT / "Salidas" / "Imagen"
LOG_ROOT = PROJECT_ROOT / "logs" / "glomerule_analysis"
TILE_SIZE = 1536
STRIDE = 1536
EDGE_THRESHOLD = 50
CSV_HEADER = [
    "file_name",
    "glomeruli_id",
    "x_min_local",
    "y_min_local",
    "x_max_local",
    "y_max_local",
    "x_min_global",
    "y_min_global",
    "x_max_global",
    "y_max_global",
    "touching_edge",
    "adjacent_tiles_used",
    "classification",
    "confidence",
    "timestamp",
    "notes",
]
SUMMARY_HEADER = [
    "base_name",
    "status",
    "selected_tiles",
    "logged_rows",
    "non_proliferative",
    "proliferative",
    "sclerosed",
    "excluded",
    "last_update",
]


@dataclass(frozen=True)
class CandidateTile:
    file_name: str
    x: int
    y: int
    x_end: int
    y_end: int
    sat_frac: float


@dataclass(frozen=True)
class Detection:
    bbox_local: Tuple[int, int, int, int]
    classification: str
    confidence: float
    notes: str
    touching_edge: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def integral_image(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")


def box_mean(ii: np.ndarray, window: int) -> np.ndarray:
    sums = ii[window:, window:] - ii[:-window, window:] - ii[window:, :-window] + ii[:-window, :-window]
    return sums / float(window * window)


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, limit: int = TILE_SIZE) -> Tuple[int, int, int, int]:
    x1 = max(0, min(limit - 1, x1))
    y1 = max(0, min(limit - 1, y1))
    x2 = max(x1 + 1, min(limit, x2))
    y2 = max(y1 + 1, min(limit, y2))
    return x1, y1, x2, y2


def load_candidate_tiles(candidate_csv: Path) -> List[CandidateTile]:
    rows: List[CandidateTile] = []
    with candidate_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                CandidateTile(
                    file_name=row["file_name"],
                    x=int(row["x"]),
                    y=int(row["y"]),
                    x_end=int(row["x_end"]),
                    y_end=int(row["y_end"]),
                    sat_frac=float(row["screen_metric_sat_frac"]),
                )
            )
    return rows


def spatial_nms(candidates: Sequence[CandidateTile], radius_tiles: int = 2) -> List[CandidateTile]:
    """
    Selecciona tiles representantes usando el barrido inicial como heatmap.
    El radio=2 reduce duplicados de un mismo glomérulo repartido entre varios tiles vecinos.
    """
    remaining = sorted(candidates, key=lambda c: (c.sat_frac, -c.y, -c.x), reverse=True)
    kept: List[CandidateTile] = []
    suppression_distance = STRIDE * radius_tiles

    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        filtered: List[CandidateTile] = []
        for other in remaining:
            dx = abs(other.x - current.x)
            dy = abs(other.y - current.y)
            if max(dx, dy) <= suppression_distance:
                continue
            filtered.append(other)
        remaining = filtered

    kept.sort(key=lambda c: (c.y, c.x, c.file_name))
    return kept


def preprocess_tile(tile_path: Path, small_size: int = 192) -> Dict[str, np.ndarray]:
    img = Image.open(tile_path).convert("RGB")
    small = img.resize((small_size, small_size), Image.Resampling.LANCZOS)

    rgb = np.asarray(small).astype(np.float32) / 255.0
    hsv = np.asarray(small.convert("HSV")).astype(np.float32) / 255.0

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    tissue = ((sat > 0.10) & (val < 0.97)).astype(np.float32)

    gray_small = small.convert("L")
    gray = np.asarray(gray_small).astype(np.float32) / 255.0
    blur_mid = np.asarray(gray_small.filter(ImageFilter.GaussianBlur(radius=3))).astype(np.float32) / 255.0
    blur_wide = np.asarray(gray_small.filter(ImageFilter.GaussianBlur(radius=7))).astype(np.float32) / 255.0

    # PAS suele realzar membranas basales y matriz mesangial en magenta/púrpura.
    purple = np.clip((((rgb[:, :, 0] + rgb[:, :, 2]) * 0.5) - rgb[:, :, 1]) * 1.8, 0.0, 1.0)
    texture = np.clip((np.abs(gray - blur_mid) + np.abs(blur_mid - blur_wide)) * 2.2, 0.0, 1.0)
    bright_lumen = np.clip((val - 0.72) * 2.8, 0.0, 1.0) * (1.0 - sat)
    dark_dense = np.clip((0.48 - gray) * 2.2, 0.0, 1.0) * np.clip((sat - 0.18) * 2.0, 0.0, 1.0)

    # Score de localización: tufo texturizado PAS + algo de luces internas.
    score = (0.33 * sat + 0.27 * purple + 0.22 * texture + 0.18 * bright_lumen) * tissue

    return {
        "img_size": np.array([img.width, img.height], dtype=np.int32),
        "small_size": np.array([small_size, small_size], dtype=np.int32),
        "sat": sat,
        "val": val,
        "tissue": tissue,
        "purple": purple,
        "texture": texture,
        "bright_lumen": bright_lumen,
        "dark_dense": dark_dense,
        "score": score,
        "rgb": rgb,
    }


def locate_best_bbox(features: Dict[str, np.ndarray]) -> Tuple[int, int, int, int, Dict[str, float]]:
    score = features["score"]
    tissue = features["tissue"]
    bright = features["bright_lumen"]
    purple = features["purple"]
    texture = features["texture"]
    small_size = int(features["small_size"][0])

    score_ii = integral_image(score)
    tissue_ii = integral_image(tissue)
    bright_ii = integral_image(bright * tissue)
    purple_ii = integral_image(purple * tissue)
    texture_ii = integral_image(texture * tissue)

    best: Tuple[float, Tuple[int, int, int, int], Dict[str, float]] | None = None

    # Rango pensado para glomérulos que caben dentro de un tile de 1536 px.
    for window in (28, 36, 44, 52, 60, 72):
        core = box_mean(score_ii, window)
        cov = box_mean(tissue_ii, window)
        lum = box_mean(bright_ii, window)
        pur = box_mean(purple_ii, window)
        txt = box_mean(texture_ii, window)

        # Sesgo suave al centro para favorecer tiles representantes después del NMS.
        centers = (np.arange(small_size - window + 1) + (window / 2.0)) / float(max(1, small_size - 1))
        centers = centers * 2.0 - 1.0
        local_cx, local_cy = np.meshgrid(centers, centers)
        center_bias = 1.0 - np.clip(np.sqrt(local_cx**2 + local_cy**2), 0.0, 1.0)

        window_score = (
            (0.42 * core) +
            (0.22 * lum) +
            (0.20 * txt) +
            (0.16 * pur) +
            (0.10 * center_bias)
        )
        valid = cov > 0.42
        if not np.any(valid):
            continue

        masked_score = np.where(valid, window_score, -np.inf)
        flat_idx = int(np.argmax(masked_score))
        best_score = float(masked_score.flat[flat_idx])
        if not math.isfinite(best_score):
            continue

        y, x = np.unravel_index(flat_idx, masked_score.shape)
        stats = {
            "window_score": best_score,
            "coverage": float(cov[y, x]),
            "lumen": float(lum[y, x]),
            "purple": float(pur[y, x]),
            "texture": float(txt[y, x]),
        }

        if best is None or best_score > best[0]:
            best = (best_score, (x, y, x + window, y + window), stats)

    if best is None:
        default = (small_size // 4, small_size // 4, 3 * small_size // 4, 3 * small_size // 4)
        return scale_bbox(default, small_size, TILE_SIZE), {
            "window_score": 0.0,
            "coverage": 0.0,
            "lumen": 0.0,
            "purple": 0.0,
            "texture": 0.0,
        }

    bbox_small = best[1]
    return scale_bbox(bbox_small, small_size, TILE_SIZE), best[2]


def scale_bbox(bbox: Tuple[int, int, int, int], src_size: int, dst_size: int) -> Tuple[int, int, int, int]:
    scale = dst_size / float(src_size)
    x1, y1, x2, y2 = bbox
    return clamp_bbox(
        int(round(x1 * scale)),
        int(round(y1 * scale)),
        int(round(x2 * scale)),
        int(round(y2 * scale)),
        dst_size,
    )


def crop_stats(features: Dict[str, np.ndarray], bbox_local: Tuple[int, int, int, int]) -> Dict[str, float]:
    small_size = int(features["small_size"][0])
    x1, y1, x2, y2 = scale_bbox(bbox_local, TILE_SIZE, small_size)

    tissue = features["tissue"][y1:y2, x1:x2]
    sat = features["sat"][y1:y2, x1:x2]
    val = features["val"][y1:y2, x1:x2]
    purple = features["purple"][y1:y2, x1:x2]
    texture = features["texture"][y1:y2, x1:x2]
    bright_lumen = features["bright_lumen"][y1:y2, x1:x2]
    dark_dense = features["dark_dense"][y1:y2, x1:x2]

    tissue_cov = float(tissue.mean()) if tissue.size else 0.0
    mask = tissue > 0
    if np.any(mask):
        sat_m = float(sat[mask].mean())
        val_m = float(val[mask].mean())
        purple_m = float(purple[mask].mean())
        texture_m = float(texture[mask].mean())
        lumen_f = float((bright_lumen[mask] > 0.18).mean())
        dark_f = float((dark_dense[mask] > 0.12).mean())
        std_f = float(purple[mask].std())
    else:
        sat_m = val_m = purple_m = texture_m = lumen_f = dark_f = std_f = 0.0

    width = max(1, bbox_local[2] - bbox_local[0])
    height = max(1, bbox_local[3] - bbox_local[1])
    aspect_ratio = max(width / height, height / width)

    touching_edge = (
        bbox_local[0] <= EDGE_THRESHOLD or
        bbox_local[1] <= EDGE_THRESHOLD or
        bbox_local[2] >= TILE_SIZE - EDGE_THRESHOLD or
        bbox_local[3] >= TILE_SIZE - EDGE_THRESHOLD
    )

    return {
        "tissue_cov": tissue_cov,
        "sat_mean": sat_m,
        "val_mean": val_m,
        "purple_mean": purple_m,
        "texture_mean": texture_m,
        "lumen_frac": lumen_f,
        "dark_frac": dark_f,
        "purple_std": std_f,
        "aspect_ratio": float(aspect_ratio),
        "touching_edge": bool(touching_edge),
    }


def classify_glomerulus(window_stats: Dict[str, float], region_stats: Dict[str, float], tile_sat_frac: float) -> Tuple[str, float, str]:
    coverage = region_stats["tissue_cov"]
    aspect_ratio = region_stats["aspect_ratio"]
    lumen = region_stats["lumen_frac"]
    dark = region_stats["dark_frac"]
    texture = region_stats["texture_mean"]
    purple_mean = region_stats["purple_mean"]
    touching_edge = region_stats["touching_edge"]
    score = window_stats["window_score"]

    if coverage < 0.32 or aspect_ratio > 2.2:
        return "Excluded", 0.58, "Glomérulo incompleto por técnica de escaneo o corte tangencial, sin estructura morfológica visible (Clase Extra: Excluyente)."

    if touching_edge and coverage < 0.48 and score < 0.30:
        return "Excluded", 0.60, "Glomérulo incompleto periférico tocando borde, estructura no visible completamente (Clase Extra: Excluyente)."

    # Criterios ISN/RPS 2018 (Extraídos del PDF de la Clínica de la Costa / Uninorte):
    # - Clase VI (Esclerosante): Glomeruloesclerosis global, daño renal crónico irreversible.
    # - Clases III/IV (Proliferativo): Lesiones proliferativas focales (< 50%) o difusas (> 50%).
    # - Clases I/II/V (No Proliferativo): Mesangial mínima, proliferación mesangial leve o membranosa (engrosamiento capilar).
    
    if dark > 0.88 and texture < 0.14 and region_stats["purple_std"] < 0.06:
        confidence = min(0.89, 0.62 + 0.60 * dark)
        return "Sclerosed", confidence, "Glomeruloesclerosis global, tufo colapsado indicando daño renal crónico irreversible (Clase VI)."

    if dark > 0.82 and texture > 0.22:
        confidence = min(0.90, 0.60 + 0.75 * max(dark - 0.70, texture))
        return "Proliferative", confidence, "Lesiones proliferativas con hipercelularidad, compatible con patrón focal o difuso (Clase III o IV)."

    if score < 0.22 and tile_sat_frac < 0.62:
        return "Excluded", 0.57, "Artefacto o tejido no glomerular, patrón no definido (Clase Extra: Excluyente)."

    confidence = min(0.88, 0.58 + 0.30 * max(purple_mean, texture) + 0.08 * tile_sat_frac)
    return "Non-Proliferative", confidence, "Alteraciones mesangiales o membranosas sin hipercelularidad marcada (Clase I, II o V)."


def edge_directions(bbox_local: Tuple[int, int, int, int]) -> List[str]:
    x1, y1, x2, y2 = bbox_local
    directions: List[str] = []
    if y1 <= EDGE_THRESHOLD:
        directions.append("TOP")
    if y2 >= TILE_SIZE - EDGE_THRESHOLD:
        directions.append("BOTTOM")
    if x1 <= EDGE_THRESHOLD:
        directions.append("LEFT")
    if x2 >= TILE_SIZE - EDGE_THRESHOLD:
        directions.append("RIGHT")
    return directions


def tile_to_row(base_name: str, tile: CandidateTile, glom_index: int, det: Detection, timestamp: str) -> List[str]:
    x1, y1, x2, y2 = det.bbox_local
    directions = edge_directions(det.bbox_local)
    return [
        tile.file_name,
        f"{base_name}_G{glom_index:05d}",
        str(x1),
        str(y1),
        str(x2),
        str(y2),
        str(tile.x + x1),
        str(tile.y + y1),
        str(tile.x + x2),
        str(tile.y + y2),
        "YES" if det.touching_edge else "NO",
        json.dumps(directions, ensure_ascii=False),
        det.classification,
        f"{det.confidence:.3f}",
        timestamp,
        det.notes,
    ]


def process_tile(tile_path: Path, tile: CandidateTile) -> Detection:
    features = preprocess_tile(tile_path)
    bbox_local, window_stats = locate_best_bbox(features)
    region_stats = crop_stats(features, bbox_local)
    label, confidence, note = classify_glomerulus(window_stats, region_stats, tile.sat_frac)
    touching_edge = region_stats["touching_edge"]

    if label == "Excluded" and not touching_edge and tile.sat_frac > 0.80:
        # Candidato muy fuerte pero ambiguo: mantener un ROI algo más amplio para revisión posterior.
        cx = (bbox_local[0] + bbox_local[2]) // 2
        cy = (bbox_local[1] + bbox_local[3]) // 2
        half = 320
        bbox_local = clamp_bbox(cx - half, cy - half, cx + half, cy + half)

    note = (
        f"Heurística fase 1. sat_frac={tile.sat_frac:.3f}; "
        f"window_score={window_stats['window_score']:.3f}; {note}"
    )
    return Detection(
        bbox_local=bbox_local,
        classification=label,
        confidence=round(confidence, 3),
        notes=note,
        touching_edge=touching_edge,
    )


def write_csv(path: Path, rows: Iterable[Sequence[str]]) -> int:
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow(list(row))
            count += 1
    return count


def update_status_and_checkpoint(
    base_name: str,
    model_name: str,
    log_dir: Path,
    selected_tiles: int,
    rows: List[List[str]],
) -> None:
    timestamp = utc_now()
    counts = {
        "Non-Proliferative": 0,
        "Proliferative": 0,
        "Sclerosed": 0,
        "Excluded": 0,
    }
    for row in rows:
        counts[row[12]] += 1

    candidate_csv = log_dir / f"{base_name}_{model_name}_candidate_tiles.csv"
    total_candidates = max(0, sum(1 for _ in candidate_csv.open(encoding="utf-8")) - 1) if candidate_csv.exists() else 0

    status_txt = log_dir / f"{base_name}_{model_name}_status.txt"
    status_txt.write_text(
        "\n".join(
            [
                f"FASE 1 - ANÁLISIS DETALLADO COMPLETADO: {base_name}",
                f"Modelo: {model_name}",
                f"Última actualización (UTC): {timestamp}",
                f"Tile dir: Salidas/Imagen/{base_name}",
                f"Tile size / stride: {TILE_SIZE} / {STRIDE}",
                f"Tiles screen-positive disponibles: {total_candidates}",
                f"Tiles representantes tras NMS: {selected_tiles}",
                f"Filas registradas en log detallado: {len(rows)}",
                "Conteo por clase:",
                f"- Non-Proliferative: {counts['Non-Proliferative']}",
                f"- Proliferative: {counts['Proliferative']}",
                f"- Sclerosed: {counts['Sclerosed']}",
                f"- Excluded: {counts['Excluded']}",
                "Criterio:",
                "- Pipeline heurístico reproducible: NMS espacial sobre candidate tiles + ROI PAS + reglas morfológicas ISN/RPS 2018 (Clases I-VI) extraídas de PDFs.",
                "- Usar estos resultados como preanotación / priorización para revisión experta.",
            ]
        ),
        encoding="utf-8",
    )

    checkpoint_path = log_dir / f"{base_name}_{model_name}_checkpoint.json"
    payload: Dict[str, object] = {}
    if checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    payload.update(
        {
            "phase": 1,
            "status": "completed_detailed_phase1_heuristic",
            "last_update": timestamp,
            "detail_log_file": str(log_dir / f"{base_name}_{model_name}_log.csv"),
            "detail_rows": len(rows),
            "selected_representative_tiles": selected_tiles,
            "heuristic_summary": {
                "non_proliferative": counts["Non-Proliferative"],
                "proliferative": counts["Proliferative"],
                "sclerosed": counts["Sclerosed"],
                "excluded": counts["Excluded"],
                "selection_rule": "candidate-tile NMS with radius=2 tiles",
                "roi_rule": "best PAS/texture/lumen ROI inside representative tile",
            },
        }
    )
    checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_global_summary(summary_rows: List[List[str]]) -> None:
    path = LOG_ROOT / "_phase1_detailed_log_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(summary_rows)


def iter_biopsies(model_name: str) -> List[str]:
    biopsies: List[str] = []
    for img_dir in sorted(IMAGE_ROOT.iterdir()):
        if not img_dir.is_dir():
            continue
        candidate_csv = LOG_ROOT / img_dir.name / model_name / f"{img_dir.name}_{model_name}_candidate_tiles.csv"
        if candidate_csv.exists():
            biopsies.append(img_dir.name)
    return biopsies


def process_biopsy(base_name: str, model_name: str, force: bool = False) -> List[str]:
    log_dir = LOG_ROOT / base_name / model_name
    candidate_csv = log_dir / f"{base_name}_{model_name}_candidate_tiles.csv"
    log_csv = log_dir / f"{base_name}_{model_name}_log.csv"

    if not candidate_csv.exists():
        return [base_name, "missing_candidates", "0", "0", "0", "0", "0", "0", utc_now()]

    if log_csv.exists() and not force:
        with log_csv.open(encoding="utf-8") as fh:
            existing_rows = max(0, sum(1 for _ in fh) - 1)
        if existing_rows > 0:
            counts = {"Non-Proliferative": 0, "Proliferative": 0, "Sclerosed": 0, "Excluded": 0}
            with log_csv.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    counts[row["classification"]] = counts.get(row["classification"], 0) + 1
            return [
                base_name,
                "skipped_existing_detail_log",
                "0",
                str(existing_rows),
                str(counts.get("Non-Proliferative", 0)),
                str(counts.get("Proliferative", 0)),
                str(counts.get("Sclerosed", 0)),
                str(counts.get("Excluded", 0)),
                utc_now(),
            ]

    candidates = load_candidate_tiles(candidate_csv)
    selected = spatial_nms(candidates, radius_tiles=2)

    rows: List[List[str]] = []
    glom_index = 1
    timestamp = utc_now()
    class_counts = {"Non-Proliferative": 0, "Proliferative": 0, "Sclerosed": 0, "Excluded": 0}

    for tile in selected:
        tile_path = IMAGE_ROOT / base_name / tile.file_name
        if not tile_path.exists():
            continue
        detection = process_tile(tile_path, tile)
        rows.append(tile_to_row(base_name, tile, glom_index, detection, timestamp))
        class_counts[detection.classification] += 1
        glom_index += 1

    log_dir.mkdir(parents=True, exist_ok=True)
    write_csv(log_csv, rows)
    update_status_and_checkpoint(base_name, model_name, log_dir, len(selected), rows)

    return [
        base_name,
        "completed_detailed_phase1_heuristic",
        str(len(selected)),
        str(len(rows)),
        str(class_counts["Non-Proliferative"]),
        str(class_counts["Proliferative"]),
        str(class_counts["Sclerosed"]),
        str(class_counts["Excluded"]),
        utc_now(),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Completa la Fase 1 detallada con heurísticas reproducibles.")
    parser.add_argument("--model-name", default="gpt4v", help="Nombre del modelo/carpeta de logs. Default: gpt4v")
    parser.add_argument("--base-name", action="append", help="Biopsia específica. Repetible.")
    parser.add_argument("--all", action="store_true", help="Procesar todas las biopsias con candidate tiles.")
    parser.add_argument("--force", action="store_true", help="Sobrescribir logs detallados existentes si ya tienen filas.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        biopsies = iter_biopsies(args.model_name)
    elif args.base_name:
        biopsies = args.base_name
    else:
        raise SystemExit("Debes indicar --all o al menos un --base-name.")

    summaries: List[List[str]] = []
    for base_name in biopsies:
        summary = process_biopsy(base_name, args.model_name, force=args.force)
        summaries.append(summary)
        print(
            f"[{summary[1]}] {base_name}: selected={summary[2]} rows={summary[3]} "
            f"NP={summary[4]} P={summary[5]} S={summary[6]} E={summary[7]}"
        )

    append_global_summary(summaries)
    print(f"\nResumen global guardado en: {LOG_ROOT / '_phase1_detailed_log_summary.csv'}")


if __name__ == "__main__":
    main()
