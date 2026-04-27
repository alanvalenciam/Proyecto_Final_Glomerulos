#!/usr/bin/env python3
"""
Fase 2 automática: validación y control de calidad (QC) post-análisis.

Objetivo:
- Auditar los `*_log.csv` producidos en Fase 1.
- Calcular estadísticas descriptivas por biopsia.
- Identificar glomérulos dudosos para re-análisis.
- Generar artefactos de QC por biopsia y un resumen global.

Notas:
- Esta implementación sigue la intención de `docs/plans/PLAN_VALIDATION_QC.md`
  pero evita dependencias no disponibles en el repo (pandas, sklearn, pyyaml).
- Si una biopsia queda con items flagged, la salida se marca como
  `pending_manual_reanalysis`: se genera el batch JSON para revisión, pero no se
  consolida un CSV final clínicamente validado hasta completar esa revisión.
- Si no hay flags y la auditoría estructural pasa, el CSV original se promueve
  automáticamente a `*_final.csv`.
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_ROOT = PROJECT_ROOT / "logs" / "glomerule_analysis"
IMAGE_ROOT = PROJECT_ROOT / "Salidas" / "Imagen"
MODEL_NAME = "gemini"
VALID_CLASSES = ["Non-Proliferative", "Proliferative", "Sclerosed", "Excluded"]
REQUIRED_COLUMNS = [
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
FLAGGED_HEADER = REQUIRED_COLUMNS + [
    "area_global",
    "width_global",
    "height_global",
    "aspect_ratio",
    "flag_reason",
    "flag_details",
]
SUMMARY_HEADER = [
    "biopsy_name",
    "model_name",
    "rows_total",
    "rows_valid_non_excluded",
    "audit_valid",
    "audit_errors",
    "audit_warnings",
    "flagged_total",
    "flagged_low_confidence",
    "flagged_atypical_size",
    "flagged_neighbor_inconsistency",
    "flagged_suspicious_morphology",
    "flagged_excluded_no_notes",
    "low_conf_threshold",
    "qc_status",
    "final_csv_created",
    "report_path",
    "statistics_path",
    "flagged_csv_path",
    "reanalysis_batch_path",
    "reanalysis_results_path",
    "concordance_path",
    "last_update",
]
TILE_COORD_RE = re.compile(r"tile_x(\d+)_y(\d+)_endx(\d+)_endy(\d+)", re.IGNORECASE)

LOW_CONF_FACTOR = 1.5
MORPH_MIN_AREA_PX2 = 500
MORPH_ASPECT_RATIO_THRESHOLD = 3.0
NEIGHBOR_STRIDE = 1536


@dataclass
class CasePaths:
    biopsy_name: str
    model_name: str
    log_csv: Path
    base_dir: Path
    image_dir: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def safe_float(value: str) -> float:
    return float(str(value).strip())


def safe_int(value: str) -> int:
    return int(float(str(value).strip()))


def load_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            serializable = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, dict)):
                    serializable[key] = json.dumps(value, ensure_ascii=False)
                else:
                    serializable[key] = value
            writer.writerow(serializable)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def parse_adjacent_tiles(value: str) -> List[str]:
    text = (value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return []


def parse_tile_coords(file_name: str) -> Optional[Tuple[int, int, int, int]]:
    match = TILE_COORD_RE.search(file_name or "")
    if not match:
        return None
    return tuple(int(group) for group in match.groups())


def composite_key(row: Dict[str, object]) -> str:
    return f"{row.get('file_name', '')}::{row.get('glomeruli_id', '')}"


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile() requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def iqr_bounds(values: Sequence[float]) -> Optional[Tuple[float, float, float, float]]:
    if len(values) < 4:
        return None
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    iqr = q3 - q1
    return q1, q3, q1 - 1.5 * iqr, q3 + 1.5 * iqr


def safe_kappa(labels_a: Sequence[str], labels_b: Sequence[str], labels: Sequence[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("Sequences must have same length for kappa")
    n = len(labels_a)
    if n == 0:
        return 1.0

    confusion = [[0 for _ in labels] for _ in labels]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    for a, b in zip(labels_a, labels_b):
        if a not in label_to_idx or b not in label_to_idx:
            continue
        confusion[label_to_idx[a]][label_to_idx[b]] += 1

    observed = sum(confusion[i][i] for i in range(len(labels))) / n
    row_marginals = [sum(row) for row in confusion]
    col_marginals = [sum(confusion[row][col] for row in range(len(labels))) for col in range(len(labels))]
    expected = sum((row_marginals[i] / n) * (col_marginals[i] / n) for i in range(len(labels)))
    if math.isclose(1.0 - expected, 0.0):
        return 1.0
    return (observed - expected) / (1.0 - expected)


def augment_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for row in rows:
        item: Dict[str, object] = dict(row)
        item["x_min_local"] = safe_int(row["x_min_local"])
        item["y_min_local"] = safe_int(row["y_min_local"])
        item["x_max_local"] = safe_int(row["x_max_local"])
        item["y_max_local"] = safe_int(row["y_max_local"])
        item["x_min_global"] = safe_int(row["x_min_global"])
        item["y_min_global"] = safe_int(row["y_min_global"])
        item["x_max_global"] = safe_int(row["x_max_global"])
        item["y_max_global"] = safe_int(row["y_max_global"])
        item["confidence"] = safe_float(row["confidence"])
        item["touching_edge"] = str(row["touching_edge"]).strip()
        item["notes"] = row.get("notes", "") or ""
        item["adjacent_tiles_used"] = row.get("adjacent_tiles_used", "[]") or "[]"
        item["adjacent_tiles_used_list"] = parse_adjacent_tiles(str(item["adjacent_tiles_used"]))
        item["width_global"] = int(item["x_max_global"]) - int(item["x_min_global"])
        item["height_global"] = int(item["y_max_global"]) - int(item["y_min_global"])
        item["area_global"] = int(item["width_global"]) * int(item["height_global"])
        h = int(item["height_global"])
        item["aspect_ratio"] = (int(item["width_global"]) / h) if h else float("inf")
        item["composite_key"] = composite_key(item)
        item["tile_coords"] = parse_tile_coords(str(item["file_name"]))
        enriched.append(item)
    return enriched


def audit_case(
    fieldnames: Sequence[str],
    rows: List[Dict[str, object]],
    image_dir: Path,
) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []

    missing_cols = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
    if missing_cols:
        errors.append(f"Columnas faltantes: {missing_cols}")

    duplicates = Counter(row["composite_key"] for row in rows)
    duplicate_keys = [key for key, count in duplicates.items() if count > 1]
    if duplicate_keys:
        errors.append(f"Duplicados detectados: {len(duplicate_keys)} claves compuestas")

    invalid_conf = [row["composite_key"] for row in rows if not (0.0 <= float(row["confidence"]) <= 1.0)]
    if invalid_conf:
        errors.append(f"Confidence fuera de rango [0,1]: {len(invalid_conf)} filas")

    invalid_bbox_local = [
        row["composite_key"]
        for row in rows
        if int(row["x_min_local"]) >= int(row["x_max_local"]) or int(row["y_min_local"]) >= int(row["y_max_local"])
    ]
    if invalid_bbox_local:
        errors.append(f"Bbox local inválido: {len(invalid_bbox_local)} filas")

    invalid_bbox_global = [
        row["composite_key"]
        for row in rows
        if int(row["x_min_global"]) >= int(row["x_max_global"]) or int(row["y_min_global"]) >= int(row["y_max_global"])
    ]
    if invalid_bbox_global:
        errors.append(f"Bbox global inválido: {len(invalid_bbox_global)} filas")

    invalid_classes = sorted({str(row["classification"]) for row in rows if row["classification"] not in VALID_CLASSES})
    if invalid_classes:
        errors.append(f"Clasificaciones inválidas: {invalid_classes}")

    invalid_touching = [row["composite_key"] for row in rows if row["touching_edge"] not in {"YES", "NO"}]
    if invalid_touching:
        errors.append(f"touching_edge debe ser YES/NO: {len(invalid_touching)} filas")

    bad_timestamps = []
    for row in rows:
        try:
            parse_iso8601(str(row["timestamp"]))
        except Exception:
            bad_timestamps.append(row["composite_key"])
    if bad_timestamps:
        errors.append(f"Timestamps inválidos: {len(bad_timestamps)} filas")

    touching_without_adj = [
        row["composite_key"]
        for row in rows
        if row["touching_edge"] == "YES" and not row["adjacent_tiles_used_list"]
    ]
    if touching_without_adj:
        warnings.append(f"Tiles tocando borde sin vecinos registrados: {len(touching_without_adj)} filas")

    missing_tiles = 0
    for row in rows:
        if not (image_dir / str(row["file_name"])).exists():
            missing_tiles += 1
    if missing_tiles:
        warnings.append(f"Tiles faltantes en disco respecto al CSV: {missing_tiles}")

    total_glomeruli = len(rows)
    total_tiles = len({row["file_name"] for row in rows})
    total_excluded = sum(1 for row in rows if row["classification"] == "Excluded")
    total_valid = total_glomeruli - total_excluded
    edge_cases = sum(1 for row in rows if row["touching_edge"] == "YES")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "total_glomeruli": total_glomeruli,
            "total_tiles": total_tiles,
            "total_excluded": total_excluded,
            "total_valid": total_valid,
            "edge_cases": edge_cases,
        },
    }


def statistics_case(rows: List[Dict[str, object]]) -> Dict[str, object]:
    valid_rows = [row for row in rows if row["classification"] != "Excluded"]

    if not valid_rows:
        return {
            "class_distribution": {},
            "confidence_stats": None,
            "low_conf_threshold": None,
            "low_conf_keys": [],
            "size_outlier_keys": [],
            "size_iqr_bounds": None,
            "area_mean_by_class": {},
            "edge_summary": {
                "touching_edge_total": sum(1 for row in rows if row["touching_edge"] == "YES"),
                "touching_edge_reconstructed": sum(
                    1 for row in rows if row["touching_edge"] == "YES" and row["adjacent_tiles_used_list"]
                ),
            },
        }

    class_distribution = Counter(str(row["classification"]) for row in valid_rows)
    confidences = [float(row["confidence"]) for row in valid_rows]
    conf_mean = mean(confidences)
    conf_median = median(confidences)
    conf_std = stdev(confidences) if len(confidences) > 1 else 0.0
    conf_min = min(confidences)
    conf_max = max(confidences)
    low_conf_threshold = max(0.0, conf_mean - LOW_CONF_FACTOR * conf_std)
    low_conf_keys = [row["composite_key"] for row in valid_rows if float(row["confidence"]) < low_conf_threshold]

    areas = [float(row["area_global"]) for row in valid_rows]
    bounds = iqr_bounds(areas)
    if bounds is None:
        size_outlier_keys: List[str] = []
        size_iqr_bounds = None
    else:
        q1, q3, lower, upper = bounds
        size_iqr_bounds = {"q1": q1, "q3": q3, "lower": lower, "upper": upper}
        size_outlier_keys = [
            row["composite_key"]
            for row in valid_rows
            if float(row["area_global"]) < lower or float(row["area_global"]) > upper
        ]

    area_mean_by_class = {}
    for label in ["Non-Proliferative", "Proliferative", "Sclerosed"]:
        subset = [float(row["area_global"]) for row in valid_rows if row["classification"] == label]
        if subset:
            area_mean_by_class[label] = mean(subset)

    touching_edge = [row for row in rows if row["touching_edge"] == "YES"]
    return {
        "class_distribution": dict(class_distribution),
        "confidence_stats": {
            "mean": conf_mean,
            "median": conf_median,
            "std": conf_std,
            "min": conf_min,
            "max": conf_max,
        },
        "low_conf_threshold": low_conf_threshold,
        "low_conf_keys": low_conf_keys,
        "size_outlier_keys": size_outlier_keys,
        "size_iqr_bounds": size_iqr_bounds,
        "area_mean_by_class": area_mean_by_class,
        "edge_summary": {
            "touching_edge_total": len(touching_edge),
            "touching_edge_reconstructed": sum(1 for row in touching_edge if row["adjacent_tiles_used_list"]),
        },
    }


def neighbor_inconsistency_keys(rows: List[Dict[str, object]]) -> List[str]:
    rows_by_tile: Dict[Tuple[int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        coords = row["tile_coords"]
        if coords is None or row["classification"] == "Excluded":
            continue
        x, y, _, _ = coords
        rows_by_tile[(x, y)].append(row)

    flagged: List[str] = []
    for row in rows:
        coords = row["tile_coords"]
        if coords is None or row["classification"] == "Excluded":
            continue
        x, y, _, _ = coords
        neighbor_classes: List[str] = []
        for dx, dy in [(-NEIGHBOR_STRIDE, 0), (NEIGHBOR_STRIDE, 0), (0, -NEIGHBOR_STRIDE), (0, NEIGHBOR_STRIDE)]:
            for other in rows_by_tile.get((x + dx, y + dy), []):
                neighbor_classes.append(str(other["classification"]))
        if len(neighbor_classes) < 2:
            continue
        votes = Counter(neighbor_classes)
        most_common_class, votes_count = votes.most_common(1)[0]
        if str(row["classification"]) != most_common_class and votes_count / len(neighbor_classes) >= 0.67:
            flagged.append(str(row["composite_key"]))
    return flagged


def identify_flagged(rows: List[Dict[str, object]], stats: Dict[str, object]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    reasons_by_key: Dict[str, List[str]] = defaultdict(list)
    details_by_key: Dict[str, List[str]] = defaultdict(list)
    counts = {
        "LOW_CONFIDENCE": 0,
        "ATYPICAL_SIZE": 0,
        "NEIGHBOR_INCONSISTENCY": 0,
        "SUSPICIOUS_MORPHOLOGY": 0,
        "EXCLUDED_NO_NOTES": 0,
    }

    low_conf_keys = set(stats.get("low_conf_keys") or [])
    for key in low_conf_keys:
        reasons_by_key[key].append("LOW_CONFIDENCE")
        threshold = stats.get("low_conf_threshold")
        if threshold is not None:
            details_by_key[key].append(f"Confianza < {threshold:.3f}")
        counts["LOW_CONFIDENCE"] += 1

    size_outlier_keys = set(stats.get("size_outlier_keys") or [])
    for key in size_outlier_keys:
        reasons_by_key[key].append("ATYPICAL_SIZE")
        details_by_key[key].append("Tamaño fuera de IQR")
        counts["ATYPICAL_SIZE"] += 1

    inconsistent_keys = set(neighbor_inconsistency_keys(rows))
    for key in inconsistent_keys:
        reasons_by_key[key].append("NEIGHBOR_INCONSISTENCY")
        details_by_key[key].append("Clase diferente a vecinos")
        counts["NEIGHBOR_INCONSISTENCY"] += 1

    for row in rows:
        key = str(row["composite_key"])
        aspect_ratio = float(row["aspect_ratio"])
        inv_aspect_ratio = (1.0 / aspect_ratio) if aspect_ratio not in {0.0, float("inf")} else float("inf")
        if (
            int(row["area_global"]) < MORPH_MIN_AREA_PX2
            or aspect_ratio > MORPH_ASPECT_RATIO_THRESHOLD
            or inv_aspect_ratio > MORPH_ASPECT_RATIO_THRESHOLD
        ):
            reasons_by_key[key].append("SUSPICIOUS_MORPHOLOGY")
            details_by_key[key].append("Morfología atípica detectada")
            counts["SUSPICIOUS_MORPHOLOGY"] += 1
        if row["classification"] == "Excluded" and not str(row["notes"]).strip():
            reasons_by_key[key].append("EXCLUDED_NO_NOTES")
            details_by_key[key].append("Excluido sin justificación")
            counts["EXCLUDED_NO_NOTES"] += 1

    flagged_rows: List[Dict[str, object]] = []
    for row in rows:
        key = str(row["composite_key"])
        reasons = reasons_by_key.get(key)
        if not reasons:
            continue
        out = dict(row)
        out["flag_reason"] = ";".join(dict.fromkeys(reasons))
        out["flag_details"] = "; ".join(dict.fromkeys(details_by_key.get(key, [])))
        flagged_rows.append(out)

    return flagged_rows, counts


def prepare_reanalysis_batch(
    case: CasePaths,
    flagged_rows: List[Dict[str, object]],
    batch_path: Path,
) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in flagged_rows:
        grouped[str(row["file_name"])].append(row)

    payload = {
        "session_id": f"{case.biopsy_name}_{case.model_name}_reanalysis_{utc_now()}",
        "biopsy": case.biopsy_name,
        "model": case.model_name,
        "total_items": len(flagged_rows),
        "tiles_to_review": len(grouped),
        "items": [],
    }

    for tile_name in sorted(grouped):
        tile_rows = grouped[tile_name]
        tile_info = {
            "file_name": tile_name,
            "tile_path": str(case.image_dir / tile_name),
            "glomeruli_to_review": [],
        }
        for row in tile_rows:
            tile_info["glomeruli_to_review"].append(
                {
                    "glomeruli_id": row["glomeruli_id"],
                    "original_classification": row["classification"],
                    "original_confidence": row["confidence"],
                    "flag_reason": row["flag_reason"],
                    "flag_details": row["flag_details"],
                    "bbox_local": {
                        "x_min": row["x_min_local"],
                        "y_min": row["y_min_local"],
                        "x_max": row["x_max_local"],
                        "y_max": row["y_max_local"],
                    },
                    "bbox_global": {
                        "x_min": row["x_min_global"],
                        "y_min": row["y_min_global"],
                        "x_max": row["x_max_global"],
                        "y_max": row["y_max_global"],
                    },
                    "touching_edge": row["touching_edge"],
                    "adjacent_tiles_used": row["adjacent_tiles_used_list"],
                    "previous_notes": row["notes"],
                }
            )
        payload["items"].append(tile_info)

    write_json(batch_path, payload)
    return payload


def create_placeholder_reanalysis_results(
    case: CasePaths,
    flagged_rows: List[Dict[str, object]],
    results_path: Path,
) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in flagged_rows:
        grouped[str(row["file_name"])].append(row)

    payload = {
        "session_id": f"{case.biopsy_name}_{case.model_name}_reanalysis_placeholder_{utc_now()}",
        "biopsy": case.biopsy_name,
        "model": case.model_name,
        "status": "not_reanalyzed",
        "pending_manual_confirmation": bool(flagged_rows),
        "total_items": len(flagged_rows),
        "tiles_to_review": len(grouped),
        "items": [],
    }

    for tile_name in sorted(grouped):
        tile_payload = {"file_name": tile_name, "glomeruli_results": []}
        for row in grouped[tile_name]:
            tile_payload["glomeruli_results"].append(
                {
                    "glomeruli_id": row["glomeruli_id"],
                    "reanalysis_classification": row["classification"],
                    "reanalysis_confidence": row["confidence"],
                    "changed": False,
                    "resolution_status": "pending_manual_confirmation",
                    "justification": (
                        "Fase 2 finalizada sin reanálisis externo; se conserva la clasificación original "
                        "y el glomérulo permanece listado en flagged_items.csv para revisión posterior."
                    ),
                    "original_flag_reason": row["flag_reason"],
                    "original_flag_details": row["flag_details"],
                }
            )
        payload["items"].append(tile_payload)

    write_json(results_path, payload)
    return payload


def promote_original_to_final(case: CasePaths) -> Tuple[bool, Path, Path]:
    final_csv_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_final.csv"
    backup_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_pre_reanalysis_backup.csv"
    copy_file(case.log_csv, backup_path)
    copy_file(case.log_csv, final_csv_path)
    return True, final_csv_path, backup_path


def concordance_from_same_source(
    case: CasePaths,
    concordance_path: Path,
    rows: List[Dict[str, object]],
) -> Dict[str, object]:
    labels = ["Non-Proliferative", "Proliferative", "Sclerosed", "Excluded"]
    confusion = [[0 for _ in labels] for _ in labels]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for row in rows:
        label = str(row["classification"])
        if label in label_to_idx:
            idx = label_to_idx[label]
            confusion[idx][idx] += 1
    total = len(rows)
    payload = {
        "model": case.model_name,
        "agreement": total,
        "total": total,
        "agreement_pct": 100.0 if total or total == 0 else 0.0,
        "kappa": 1.0,
        "confusion_matrix": confusion,
        "status": "identity_copy_no_reanalysis_needed",
        "timestamp": utc_now(),
    }
    write_json(concordance_path, payload)
    return payload


def apply_reanalysis_and_concordance(
    case: CasePaths,
    rows: List[Dict[str, object]],
    reanalysis_results: Dict[str, object],
    fieldnames: Sequence[str],
) -> Tuple[Path, Path, Dict[str, object]]:
    updated_rows = [dict(r) for r in rows]
    lookup = {str(r["glomeruli_id"]): r for r in updated_rows}
    
    for item in reanalysis_results.get("items", []):
        for glom in item.get("glomeruli_results", []):
            gid = str(glom["glomeruli_id"])
            if gid in lookup:
                row = lookup[gid]
                row["classification"] = glom.get("reanalysis_classification", row["classification"])
                if "reanalysis_confidence" in glom:
                    row["confidence"] = glom["reanalysis_confidence"]
                if glom.get("justification"):
                    row["notes"] = f"[REANALYZED] {glom['justification']} | {row.get('notes', '')}".strip(" |")
    
    labels = ["Non-Proliferative", "Proliferative", "Sclerosed", "Excluded"]
    confusion = [[0 for _ in labels] for _ in labels]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    orig_labels = [str(r["classification"]) for r in rows]
    new_labels = [str(r["classification"]) for r in updated_rows]
    
    for a, b in zip(orig_labels, new_labels):
        if a in label_to_idx and b in label_to_idx:
            confusion[label_to_idx[a]][label_to_idx[b]] += 1
            
    total = len(orig_labels)
    agreement = sum(confusion[i][i] for i in range(len(labels)))
    kappa = safe_kappa(orig_labels, new_labels, labels)
    
    concordance = {
        "model": case.model_name,
        "agreement": agreement,
        "total": total,
        "agreement_pct": (agreement / total * 100.0) if total else 0.0,
        "kappa": kappa,
        "confusion_matrix": confusion,
        "status": "reanalyzed",
        "timestamp": utc_now(),
    }
    
    concordance_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_concordance.json"
    write_json(concordance_path, concordance)
    
    final_csv_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_final.csv"
    backup_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_pre_reanalysis_backup.csv"
    
    copy_file(case.log_csv, backup_path)
    write_csv(final_csv_path, fieldnames, updated_rows)
    
    return final_csv_path, backup_path, concordance


def render_html_report(
    case: CasePaths,
    audit: Dict[str, object],
    stats: Dict[str, object],
    flagged_rows: List[Dict[str, object]],
    flag_counts: Dict[str, int],
    qc_status: str,
    final_csv_path: Optional[Path],
    reanalysis_batch_path: Path,
    concordance_path: Optional[Path],
    report_path: Path,
) -> None:
    audit_errors = audit["errors"]
    audit_warnings = audit["warnings"]
    class_distribution = stats.get("class_distribution", {})
    conf_stats = stats.get("confidence_stats")
    edge_summary = stats.get("edge_summary", {})

    flagged_preview = "".join(
        f"<tr><td>{html.escape(str(row['glomeruli_id']))}</td><td>{html.escape(str(row['classification']))}</td>"
        f"<td>{float(row['confidence']):.3f}</td><td>{html.escape(str(row['flag_reason']))}</td>"
        f"<td>{html.escape(str(row['flag_details']))}</td></tr>"
        for row in flagged_rows[:50]
    )
    if not flagged_preview:
        flagged_preview = "<tr><td colspan='5'>Sin items flagged</td></tr>"

    class_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
        for label, count in sorted(class_distribution.items())
    ) or "<tr><td colspan='2'>Sin glomérulos válidos</td></tr>"

    recommendations = []
    if audit_errors:
        recommendations.append("Corregir los errores estructurales antes de usar estos datos como referencia clínica.")
    if flagged_rows:
        recommendations.append("Enviar el batch de items flagged a revisión experta/LLM antes de consolidar un CSV final.")
    if not audit_errors and not flagged_rows:
        recommendations.append("El caso pasó la auditoría y no requiere re-análisis adicional en esta fase automática.")
    if audit_warnings:
        recommendations.append("Revisar manualmente los warnings para decidir si ameritan ajuste del pipeline.")

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>QC Report - {html.escape(case.biopsy_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ color: #173b70; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d4d8df; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3fb; }}
    .ok {{ color: #0a6d2a; font-weight: bold; }}
    .warn {{ color: #9a6700; font-weight: bold; }}
    .bad {{ color: #b42318; font-weight: bold; }}
    code {{ background: #f4f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Reporte QC / Fase 2</h1>
  <p><strong>Biopsia:</strong> {html.escape(case.biopsy_name)}<br>
     <strong>Modelo:</strong> {html.escape(case.model_name)}<br>
     <strong>Estado QC:</strong> <code>{html.escape(qc_status)}</code><br>
     <strong>Generado:</strong> {html.escape(utc_now())}</p>

  <h2>1. Auditoría Estructural</h2>
  <p class="{'ok' if audit['valid'] else 'bad'}">{'✓ Auditoría válida' if audit['valid'] else '✗ Auditoría con errores'}</p>
  <ul>
    <li>Glomérulos totales: {audit['summary']['total_glomeruli']}</li>
    <li>Tiles únicos: {audit['summary']['total_tiles']}</li>
    <li>Glomérulos válidos (no excluidos): {audit['summary']['total_valid']}</li>
    <li>Excluidos: {audit['summary']['total_excluded']}</li>
    <li>Edge cases: {audit['summary']['edge_cases']}</li>
  </ul>
  <p><strong>Errores:</strong> {'; '.join(audit_errors) if audit_errors else 'Ninguno'}</p>
  <p><strong>Warnings:</strong> {'; '.join(audit_warnings) if audit_warnings else 'Ninguno'}</p>

  <h2>2. Estadísticas</h2>
  <table>
    <tr><th>Clase</th><th>Conteo</th></tr>
    {class_rows}
  </table>
  <ul>
    <li>Threshold baja confianza: {f"{stats['low_conf_threshold']:.3f}" if stats.get('low_conf_threshold') is not None else 'N/A'}</li>
    <li>Touching edge: {edge_summary.get('touching_edge_total', 0)}</li>
    <li>Reconstruidos con vecinos: {edge_summary.get('touching_edge_reconstructed', 0)}</li>
    <li>Confidence mean/median/std:
      {f"{conf_stats['mean']:.3f} / {conf_stats['median']:.3f} / {conf_stats['std']:.3f}" if conf_stats else 'N/A'}
    </li>
  </ul>

  <h2>3. Flagged para re-análisis</h2>
  <ul>
    <li>Total items flagged: {len(flagged_rows)}</li>
    <li>LOW_CONFIDENCE: {flag_counts['LOW_CONFIDENCE']}</li>
    <li>ATYPICAL_SIZE: {flag_counts['ATYPICAL_SIZE']}</li>
    <li>NEIGHBOR_INCONSISTENCY: {flag_counts['NEIGHBOR_INCONSISTENCY']}</li>
    <li>SUSPICIOUS_MORPHOLOGY: {flag_counts['SUSPICIOUS_MORPHOLOGY']}</li>
    <li>EXCLUDED_NO_NOTES: {flag_counts['EXCLUDED_NO_NOTES']}</li>
  </ul>
  <p><strong>Batch JSON:</strong> <code>{html.escape(str(reanalysis_batch_path))}</code></p>
  <table>
    <tr><th>Glomeruli ID</th><th>Clase</th><th>Confidence</th><th>Flag</th><th>Detalles</th></tr>
    {flagged_preview}
  </table>

  <h2>4. Salidas</h2>
  <ul>
    <li>CSV final: <code>{html.escape(str(final_csv_path)) if final_csv_path else 'No creado (pendiente re-análisis)'}</code></li>
    <li>Concordancia: <code>{html.escape(str(concordance_path)) if concordance_path else 'No aplica todavía'}</code></li>
    <li>Reporte actual: <code>{html.escape(str(report_path))}</code></li>
  </ul>

  <h2>5. Recomendaciones</h2>
  <ul>
    {''.join(f'<li>{html.escape(item)}</li>' for item in recommendations)}
  </ul>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_content, encoding="utf-8")


def discover_cases(model_name: str) -> List[CasePaths]:
    cases: List[CasePaths] = []
    for log_csv in sorted(LOG_ROOT.glob(f"*/{model_name}/*_{model_name}_log.csv")):
        biopsy_name = log_csv.parent.parent.name
        cases.append(
            CasePaths(
                biopsy_name=biopsy_name,
                model_name=model_name,
                log_csv=log_csv,
                base_dir=log_csv.parent,
                image_dir=IMAGE_ROOT / biopsy_name,
            )
        )
    return cases


def process_case(case: CasePaths) -> Dict[str, object]:
    fieldnames, raw_rows = load_csv(case.log_csv)
    rows = augment_rows(raw_rows)

    audit = audit_case(fieldnames, rows, case.image_dir)
    stats = statistics_case(rows)
    flagged_rows, flag_counts = identify_flagged(rows, stats)

    flagged_csv_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_flagged_items.csv"
    write_csv(flagged_csv_path, FLAGGED_HEADER, flagged_rows)

    reanalysis_batch_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_reanalysis_batch.json"
    reanalysis_batch = prepare_reanalysis_batch(case, flagged_rows, reanalysis_batch_path)
    reanalysis_results_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_reanalysis_results.json"
    
    reanalysis_results = None
    if reanalysis_results_path.exists():
        with open(reanalysis_results_path, "r", encoding="utf-8") as f:
            reanalysis_results = json.load(f)
            
    if not reanalysis_results or reanalysis_results.get("status") == "not_reanalyzed":
        reanalysis_results = create_placeholder_reanalysis_results(case, flagged_rows, reanalysis_results_path)

    statistics_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_statistics.json"

    final_csv_path: Optional[Path] = None
    backup_path: Optional[Path] = None
    concordance_path: Optional[Path] = None
    concordance = None

    if not audit["valid"]:
        qc_status = "audit_failed"
    else:
        if reanalysis_results.get("status") == "completed" or (reanalysis_results.get("status") != "not_reanalyzed" and not reanalysis_results.get("pending_manual_confirmation", True)):
            qc_status = "finalized_with_reanalysis"
            final_csv_path, backup_path, concordance = apply_reanalysis_and_concordance(case, rows, reanalysis_results, fieldnames)
            concordance_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_concordance.json"
        else:
            qc_status = "finalized_with_pending_flags" if flagged_rows else "finalized_no_flags"
            _, final_csv_path, backup_path = promote_original_to_final(case)
            concordance_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_concordance.json"
            concordance = concordance_from_same_source(case, concordance_path, rows)

    statistics_payload = {
        "biopsy_name": case.biopsy_name,
        "model_name": case.model_name,
        "qc_status": qc_status,
        "log_csv": str(case.log_csv),
        "final_csv": str(final_csv_path) if final_csv_path else None,
        "backup_csv": str(backup_path) if backup_path else None,
        "audit": audit,
        "statistics": stats,
        "flag_counts": flag_counts,
        "flagged_total": len(flagged_rows),
        "reanalysis": {
            "pending": bool(flagged_rows),
            "batch_path": str(reanalysis_batch_path),
            "results_path": str(reanalysis_results_path),
            "batch_items": reanalysis_batch["total_items"],
            "tiles_to_review": reanalysis_batch["tiles_to_review"],
            "results_status": reanalysis_results["status"],
        },
        "concordance": concordance,
        "generated_at": utc_now(),
    }
    write_json(statistics_path, statistics_payload)

    report_path = case.base_dir / f"{case.biopsy_name}_{case.model_name}_QC_REPORT.html"
    render_html_report(
        case=case,
        audit=audit,
        stats=stats,
        flagged_rows=flagged_rows,
        flag_counts=flag_counts,
        qc_status=qc_status,
        final_csv_path=final_csv_path,
        reanalysis_batch_path=reanalysis_batch_path,
        concordance_path=concordance_path,
        report_path=report_path,
    )

    return {
        "biopsy_name": case.biopsy_name,
        "model_name": case.model_name,
        "rows_total": len(rows),
        "rows_valid_non_excluded": sum(1 for row in rows if row["classification"] != "Excluded"),
        "audit_valid": audit["valid"],
        "audit_errors": " | ".join(audit["errors"]),
        "audit_warnings": " | ".join(audit["warnings"]),
        "flagged_total": len(flagged_rows),
        "flagged_low_confidence": flag_counts["LOW_CONFIDENCE"],
        "flagged_atypical_size": flag_counts["ATYPICAL_SIZE"],
        "flagged_neighbor_inconsistency": flag_counts["NEIGHBOR_INCONSISTENCY"],
        "flagged_suspicious_morphology": flag_counts["SUSPICIOUS_MORPHOLOGY"],
        "flagged_excluded_no_notes": flag_counts["EXCLUDED_NO_NOTES"],
        "low_conf_threshold": (
            f"{stats['low_conf_threshold']:.6f}" if stats.get("low_conf_threshold") is not None else ""
        ),
        "qc_status": qc_status,
        "final_csv_created": bool(final_csv_path),
        "report_path": str(report_path),
        "statistics_path": str(statistics_path),
        "flagged_csv_path": str(flagged_csv_path),
        "reanalysis_batch_path": str(reanalysis_batch_path),
        "reanalysis_results_path": str(reanalysis_results_path),
        "concordance_path": str(concordance_path) if concordance_path else "",
        "last_update": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta la Fase 2 de validación/QC sobre los logs glomerulares.")
    parser.add_argument("--all", action="store_true", help="Procesa todos los `*_log.csv` de logs/glomerule_analysis.")
    parser.add_argument("--biopsy", action="append", default=[], help="Biopsia específica a procesar (repetible).")
    parser.add_argument("--model", type=str, default="gemini", help="Modelo a procesar (ej: gemini, gpt4v).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = discover_cases(args.model)
    if not cases:
        raise SystemExit("No se encontraron logs de Fase 1 para procesar.")

    selected: List[CasePaths]
    if args.all or not args.biopsy:
        selected = cases
    else:
        wanted = set(args.biopsy)
        selected = [case for case in cases if case.biopsy_name in wanted]
        missing = sorted(wanted - {case.biopsy_name for case in selected})
        if missing:
            raise SystemExit(f"Biopsias no encontradas: {missing}")

    summary_rows = [process_case(case) for case in selected]

    summary_csv = LOG_ROOT / "_phase2_validation_summary.csv"
    summary_json = LOG_ROOT / "_phase2_validation_summary.json"
    flagged_overview_csv = LOG_ROOT / "_phase2_flagged_overview.csv"

    write_csv(summary_csv, SUMMARY_HEADER, summary_rows)
    write_json(
        summary_json,
        {
            "processed_cases": len(summary_rows),
            "summary_csv": str(summary_csv),
            "generated_at": utc_now(),
            "status_counts": dict(Counter(row["qc_status"] for row in summary_rows)),
            "flagged_total": sum(int(row["flagged_total"]) for row in summary_rows),
            "cases_with_flags": [row["biopsy_name"] for row in summary_rows if int(row["flagged_total"]) > 0],
            "rows": summary_rows,
        },
    )

    flagged_rows_all: List[Dict[str, object]] = []
    for row in summary_rows:
        flagged_csv = Path(str(row["flagged_csv_path"]))
        _, flagged_part = load_csv(flagged_csv)
        for item in flagged_part:
            item["biopsy_name"] = row["biopsy_name"]
            item["model_name"] = row["model_name"]
            flagged_rows_all.append(item)
    if flagged_rows_all:
        flagged_fieldnames = ["biopsy_name", "model_name"] + FLAGGED_HEADER
    else:
        flagged_fieldnames = ["biopsy_name", "model_name"] + FLAGGED_HEADER
    write_csv(flagged_overview_csv, flagged_fieldnames, flagged_rows_all)

    print(f"Casos procesados: {len(summary_rows)}")
    print(f"Resumen CSV: {summary_csv}")
    print(f"Resumen JSON: {summary_json}")
    print(f"Flagged overview: {flagged_overview_csv}")
    for status, count in Counter(row["qc_status"] for row in summary_rows).items():
        print(f"  - {status}: {count}")
    print(f"Items flagged totales: {sum(int(row['flagged_total']) for row in summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
