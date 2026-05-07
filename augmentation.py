#!/usr/bin/env python3
"""
Data augmentation script for glomeruli detection with balanced class distribution.

Architecture:
  Componente A: Generate 6 deterministic transforms + log originals
  Analysis: Calculate synthetic targets for 37/24/21/18% distribution
  Componente B: Generate synthetic data from originals to reach targets
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import shutil
import random

import numpy as np
import cv2
from PIL import Image
import imagehash
import albumentations as A
from tqdm import tqdm


logger = logging.getLogger(__name__)


TARGET_ESCLEROSADO = 0.20
TARGET_EXCLUIDO = 0.15

MAX_AUGMENTATIONS_PER_ORIGINAL = 3
PHASH_THRESHOLD = 3
PIXEL_DIFF_THRESHOLD = 2 / 255.0

CLASS_NAMES = {
    'No_Proliferativo': 1,
    'Proliferativo': 2,
    'Esclerosado': 3,
    'Excluido': 4,
}


@dataclass
class TileEntry:
    """Metadata for a tile."""
    stem: str
    biopsia: str
    classes: Set[str]
    img_path: Path
    mask_path: Path
    tile_info: dict = None


@dataclass
class AugStats:
    """Statistics for augmentation."""
    candidates_found: int = 0
    max_possible: int = 0
    generated: int = 0
    discarded_phash: int = 0
    discarded_pixdiff: int = 0
    discarded_io_error: int = 0

def check_dependencies():
    missing = []
    try:
        import albumentations
    except ImportError:
        missing.append('albumentations>=1.3.0')

    try:
        import imagehash
    except ImportError:
        missing.append('imagehash')

    if missing:
        print("ERROR: Missing dependencies:", ', '.join(missing))
        print("Install with: pip install " + ' '.join(missing))
        sys.exit(1)


def load_image(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None


def load_mask(path: Path) -> Optional[np.ndarray]:
    try:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        return mask
    except Exception as e:
        logger.error(f"Error loading mask {path}: {e}")
        return None


def save_image(img: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def save_pair(img: np.ndarray, mask: np.ndarray, img_path: Path, mask_path: Path):
    save_image(img, img_path)
    save_mask(mask, mask_path)


def build_augmentation_pipeline() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.03, 0.03)},
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            shear=0,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.5
        ),
    ], additional_targets={'mask': 'mask'})


DETERMINISTIC_TRANSFORMS = {
    'hflip': lambda x: np.fliplr(x),
    'vflip': lambda x: np.flipud(x),
    'rot90': lambda x: np.rot90(x, k=1),
    'rot180': lambda x: np.rot90(x, k=2),
    'rot270': lambda x: np.rot90(x, k=3),
    'transpose': lambda x: np.transpose(x, (1, 0, 2)) if x.ndim == 3 else np.transpose(x),
}


def scan_dataset(tiles_root: Path) -> Tuple[List[TileEntry], Dict[str, List[TileEntry]]]:
    catalog_a = []
    catalog_b = {'Esclerosado': [], 'Excluido': []}

    for biopsia_dir in tiles_root.iterdir():
        if not biopsia_dir.is_dir():
            continue

        annotations_path = biopsia_dir / 'annotations.json'
        if not annotations_path.exists():
            logger.warning(f"No annotations.json in {biopsia_dir.name}")
            continue

        try:
            with open(annotations_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading {annotations_path}: {e}")
            continue

        for tile_info in data.get('tiles', []):
            if not tile_info.get('has_annotations'):
                continue

            img_filename = tile_info['image'].split('/')[-1]
            mask_filename = tile_info['mask'].split('/')[-1]
            stem = img_filename.replace('.png', '')

            img_path = biopsia_dir / 'images' / img_filename
            mask_path = biopsia_dir / 'masks' / mask_filename

            if not img_path.exists() or not mask_path.exists():
                continue

            classes = {g['class'] for g in tile_info.get('glomeruli', [])}

            entry = TileEntry(
                stem=stem,
                biopsia=biopsia_dir.name,
                classes=classes,
                img_path=img_path,
                mask_path=mask_path,
                tile_info=tile_info,
            )

            catalog_a.append(entry)

            if classes == {'Esclerosado'}:
                catalog_b['Esclerosado'].append(entry)
            elif classes == {'Excluido'}:
                catalog_b['Excluido'].append(entry)

    return catalog_a, catalog_b


def count_glomeruli_by_class(tiles_root: Path) -> Dict[str, int]:
    counts = {cls: 0 for cls in CLASS_NAMES.keys()}

    for biopsia_dir in tiles_root.iterdir():
        if not biopsia_dir.is_dir():
            continue

        annotations_path = biopsia_dir / 'annotations.json'
        if not annotations_path.exists():
            continue

        try:
            with open(annotations_path) as f:
                data = json.load(f)
        except Exception:
            continue

        for tile_info in data.get('tiles', []):
            for glom in tile_info.get('glomeruli', []):
                class_name = glom.get('class')
                if class_name in counts:
                    counts[class_name] += 1

    return counts


def build_augmented_tile_info(tile_info: dict, source_transform: str) -> dict:
    """Build tile_info for augmented tiles, preserving valid fields and removing invalid ones."""
    return {
        'type': 'augmented',
        'image': tile_info['image'],
        'mask': tile_info['mask'],
        'origin_native': tile_info['origin_native'],
        'has_annotations': tile_info['has_annotations'],
        'glomeruli': [dict(g) for g in tile_info.get('glomeruli', [])],
        'source_transform': source_transform,
    }


def update_annotations_json(output_root: Path, augmented_entries: Dict[str, List[dict]]):
    """Add augmented tile entries to annotations.json for each biopsia."""
    for biopsia, new_tiles in augmented_entries.items():
        if not new_tiles:
            continue

        ann_path = output_root / biopsia / 'annotations.json'
        if not ann_path.exists():
            logger.warning(f"annotations.json not found for {biopsia}, skipping")
            continue

        try:
            with open(ann_path) as f:
                ann = json.load(f)

            ann['tiles'].extend(new_tiles)

            with open(ann_path, 'w') as f:
                json.dump(ann, f, indent=2)

            logger.info(f"Updated annotations.json for {biopsia}: added {len(new_tiles)} tiles")
        except Exception as e:
            logger.error(f"Error updating annotations.json for {biopsia}: {e}")


def run_general_augmentation(
    catalog_a: List[TileEntry],
    output_root: Path,
    dry_run: bool = False,
) -> Tuple[int, List[Dict], Dict[str, List[dict]]]:
    """
    Apply 6 deterministic transforms and log originals.

    Returns:
        (number of new pairs, list of tile metadata, augmented entries by biopsia)
    """
    logger.info(f"Component A: General augmentation ({len(catalog_a)} tiles)")

    generated = 0
    tile_log = []
    augmented_entries = {}

    for entry in tqdm(catalog_a, desc="General augmentation"):
        img = load_image(entry.img_path)
        mask = load_mask(entry.mask_path)

        if img is None or mask is None:
            continue

        output_dir = output_root / entry.biopsia

        tile_log.append({
            'stem': entry.stem,
            'biopsia': entry.biopsia,
            'classes': list(entry.classes),
            'img_path': str(entry.img_path),
            'mask_path': str(entry.mask_path),
        })

        if entry.biopsia not in augmented_entries:
            augmented_entries[entry.biopsia] = []

        for transform_name, transform_fn in DETERMINISTIC_TRANSFORMS.items():
            try:
                if img.ndim == 3:
                    aug_img = transform_fn(img)
                else:
                    aug_img = transform_fn(img[..., np.newaxis])

                aug_mask = transform_fn(mask)

                unique_vals = set(np.unique(aug_mask))
                if not unique_vals.issubset({0, 255}):
                    logger.warning(f"Invalid mask values: {unique_vals}")
                    continue

                if not dry_run:
                    img_out = output_dir / 'images' / f"{entry.stem}_{transform_name}.png"
                    mask_out = output_dir / 'masks' / f"{entry.stem}_{transform_name}_mask.png"
                    save_pair(aug_img, aug_mask, img_out, mask_out)

                    aug_tile = build_augmented_tile_info(entry.tile_info, transform_name)
                    aug_tile['image'] = f"images/{entry.stem}_{transform_name}.png"
                    aug_tile['mask'] = f"masks/{entry.stem}_{transform_name}_mask.png"
                    augmented_entries[entry.biopsia].append(aug_tile)

                generated += 1

            except Exception as e:
                logger.error(f"Error applying {transform_name}: {e}")

    logger.info(f"Component A: Generated {generated} new pairs, logged {len(tile_log)} originals")
    return generated, tile_log, augmented_entries


def calculate_synthetic_targets(
    glomeruli_original: Dict[str, int],
    target_esclerosado: float,
    target_excluido: float,
    n_deterministic_transforms: int = 6,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Calculate synthetic targets for Component B based on desired distribution percentages.

    After Component A (6 deterministic transforms), all glomeruli are multiplied by 7.
    If Esclerosado or Excluido fall below their target percentages, Component B is
    triggered to synthesize additional tiles.

    Returns:
        (targets_for_B, glomeruli_after_A)
    """
    n_total = 1 + n_deterministic_transforms
    glomeruli_after_A = {c: v * n_total for c, v in glomeruli_original.items()}
    total_after_A = sum(glomeruli_after_A.values())

    needs_synthesis = {
        'Esclerosado': glomeruli_after_A['Esclerosado'] / total_after_A < target_esclerosado,
        'Excluido': glomeruli_after_A['Excluido'] / total_after_A < target_excluido,
    }

    targets = {c: 0 for c in glomeruli_original}

    if not any(needs_synthesis.values()):
        logger.info(f"No synthesis needed: Escl={glomeruli_after_A['Esclerosado']/total_after_A:.1%} (target {target_esclerosado:.0%}), "
                   f"Excl={glomeruli_after_A['Excluido']/total_after_A:.1%} (target {target_excluido:.0%})")
        return targets, glomeruli_after_A

    fixed_sum = glomeruli_after_A['No_Proliferativo'] + glomeruli_after_A['Proliferativo']
    variable_target_pct = 0.0

    for class_name, target_pct in [('Esclerosado', target_esclerosado), ('Excluido', target_excluido)]:
        if needs_synthesis[class_name]:
            variable_target_pct += target_pct
        else:
            fixed_sum += glomeruli_after_A[class_name]

    total_needed = fixed_sum / (1.0 - variable_target_pct)

    for class_name, target_pct in [('Esclerosado', target_esclerosado), ('Excluido', target_excluido)]:
        if needs_synthesis[class_name]:
            targets[class_name] = max(0, int(target_pct * total_needed) - glomeruli_after_A[class_name])

    return targets, glomeruli_after_A


def run_synthetic_generation(
    catalog_b: Dict[str, List[TileEntry]],
    output_root: Path,
    targets: Dict[str, int],
    pipeline: A.Compose,
    max_aug: int = 3,
    hash_threshold: int = 3,
    pixel_threshold: float = 2 / 255.0,
    dry_run: bool = False,
) -> Tuple[Dict[str, AugStats], Dict[str, List[dict]]]:
    """
    Generate synthetic data from ORIGINAL tiles to reach targets.

    Returns:
        (Dict mapping class name to AugStats, augmented entries by biopsia)
    """
    logger.info("Component B: Synthetic generation from original tiles")

    stats = {}
    augmented_entries = {}

    for class_name in ['Esclerosado', 'Excluido']:
        candidates = catalog_b.get(class_name, [])
        target = targets.get(class_name, 0)

        logger.info(f"  {class_name}: {len(candidates)} candidates, target {target} new")

        s = AugStats(candidates_found=len(candidates))
        s.max_possible = len(candidates) * max_aug

        if len(candidates) == 0 or target == 0:
            logger.warning(f"  No candidates or target for {class_name}")
            stats[class_name] = s
            continue

        random.shuffle(candidates)
        generated = 0
        aug_count_per_original = {}

        for entry in candidates:
            if entry.biopsia not in augmented_entries:
                augmented_entries[entry.biopsia] = []

        # Scan disk to detect existing augmented files and initialize counter
        # This prevents filename collisions on re-runs
        if len(candidates) > 0:
            sample_entry = candidates[0]
            output_dir = output_root / sample_entry.biopsia
            images_dir = output_dir / 'images'

            if images_dir.exists():
                for existing_file in images_dir.glob('*_aug*.png'):
                    parts = existing_file.stem.rsplit('_aug', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        original_stem = parts[0]
                        existing_count = int(parts[1])
                        aug_count_per_original[original_stem] = max(
                            aug_count_per_original.get(original_stem, 0),
                            existing_count
                        )

        for entry in tqdm(candidates, desc=f"{class_name} synthesis"):
            if generated >= target:
                break

            # Load ORIGINAL tile (not transformed versions)
            img = load_image(entry.img_path)
            mask = load_mask(entry.mask_path)

            if img is None or mask is None:
                s.discarded_io_error += 1
                continue

            if entry.stem not in aug_count_per_original:
                aug_count_per_original[entry.stem] = 0
            orig_phash = imagehash.phash(Image.fromarray(img))

            # Up to max_aug attempts
            for attempt in range(max_aug):
                if generated >= target:
                    break
                if aug_count_per_original[entry.stem] >= max_aug:
                    break

                try:
                    result = pipeline(image=img, mask=mask)
                    aug_img = result['image']
                    aug_mask = result['mask']

                    unique_vals = set(np.unique(aug_mask))
                    if not unique_vals.issubset({0, 255}):
                        continue

                    pixel_diff = np.mean(np.abs(aug_img.astype(float) - img.astype(float))) / 255.0
                    if pixel_diff < pixel_threshold:
                        s.discarded_pixdiff += 1
                        continue

                    aug_phash = imagehash.phash(Image.fromarray(aug_img))
                    hamming_dist = orig_phash - aug_phash
                    if hamming_dist <= hash_threshold:
                        s.discarded_phash += 1
                        continue

                    N = aug_count_per_original[entry.stem] + 1
                    aug_stem = f"{entry.stem}_aug{N}"

                    if not dry_run:
                        output_dir = output_root / entry.biopsia
                        img_out = output_dir / 'images' / f"{aug_stem}.png"
                        mask_out = output_dir / 'masks' / f"{aug_stem}_mask.png"
                        save_pair(aug_img, aug_mask, img_out, mask_out)

                        aug_tile = build_augmented_tile_info(entry.tile_info, 'random_augmentation')
                        aug_tile['image'] = f"images/{aug_stem}.png"
                        aug_tile['mask'] = f"masks/{aug_stem}_mask.png"
                        augmented_entries[entry.biopsia].append(aug_tile)

                    aug_count_per_original[entry.stem] += 1
                    generated += 1
                    s.generated += 1

                except Exception as e:
                    logger.error(f"Error augmenting {entry.stem}: {e}")

        stats[class_name] = s

    return stats, augmented_entries


def copy_base_dataset(src_root: Path, dst_root: Path, overwrite: bool = False) -> int:
    logger.info(f"Copying base dataset: {src_root} → {dst_root}")

    if dst_root.exists() and overwrite:
        logger.info(f"Removing existing {dst_root}")
        shutil.rmtree(dst_root)

    pair_count = 0

    for biopsia_dir in src_root.iterdir():
        if not biopsia_dir.is_dir():
            continue

        images_src = biopsia_dir / 'images'
        masks_src = biopsia_dir / 'masks'

        if not images_src.exists():
            continue

        images_dst = dst_root / biopsia_dir.name / 'images'
        masks_dst = dst_root / biopsia_dir.name / 'masks'

        images_dst.mkdir(parents=True, exist_ok=True)
        masks_dst.mkdir(parents=True, exist_ok=True)

        for img_file in images_src.glob('*.png'):
            dst_file = images_dst / img_file.name
            if not dst_file.exists():
                shutil.copy2(img_file, dst_file)
                pair_count += 1

        if masks_src.exists():
            for mask_file in masks_src.glob('*.png'):
                dst_file = masks_dst / mask_file.name
                if not dst_file.exists():
                    shutil.copy2(mask_file, dst_file)

        annotations_src = biopsia_dir / 'annotations.json'
        if annotations_src.exists():
            annotations_dst = dst_root / biopsia_dir.name / 'annotations.json'
            shutil.copy2(annotations_src, annotations_dst)

    logger.info(f"Base dataset copied: {pair_count} pairs")
    return pair_count


def print_report(
    n_base: int,
    n_general: int,
    glomeruli_original: Dict[str, int],
    glomeruli_after_a: Dict[str, int],
    glomeruli_targets: Dict[str, int],
    glomeruli_final: Dict[str, int],
    stats_b: Dict[str, AugStats],
    target_esclerosado: float,
    target_excluido: float,
):
    """Print detailed augmentation report."""
    print("\n" + "=" * 80)
    print("AUGMENTATION REPORT")
    print("=" * 80)

    print(f"\nBase dataset: {n_base} image-mask pairs")
    print(f"Component A: {n_general} new pairs from deterministic transforms")

    total_glomeruli_original = sum(glomeruli_original.values())
    total_glomeruli_after_a = sum(glomeruli_after_a.values())
    total_glomeruli_final = sum(glomeruli_final.values())

    print(f"\nGlomeruli distribution:")
    print(f"  Original:              {total_glomeruli_original:5}")
    print(f"  After Component A (×7):{total_glomeruli_after_a:5}")
    print(f"  Final (after B):       {total_glomeruli_final:5}")
    print()

    print(f"{'Class':<20} {'Orig':>5} {'After A':>7} {'Final':>6} {'% Final':>8} {'Target':>7}")
    print("-" * 70)

    for class_name in ['No_Proliferativo', 'Proliferativo', 'Esclerosado', 'Excluido']:
        orig = glomeruli_original.get(class_name, 0)
        after_a = glomeruli_after_a.get(class_name, 0)
        final = glomeruli_final.get(class_name, 0)
        pct_final = (final / total_glomeruli_final * 100) if total_glomeruli_final > 0 else 0

        if class_name == 'Esclerosado':
            target_str = f"{target_esclerosado:.0%}"
        elif class_name == 'Excluido':
            target_str = f"{target_excluido:.0%}"
        else:
            target_str = "—"

        print(f"{class_name:<20} {orig:>5} {after_a:>7} {final:>6} {pct_final:>7.1f}% {target_str:>7}")

    print()
    print("Synthetic generation (Component B):")
    for class_name in ['Esclerosado', 'Excluido']:
        s = stats_b.get(class_name)
        target = glomeruli_targets.get(class_name, 0)

        if target == 0:
            print(f"\n  {class_name}: ✓ Already at/above target (no synthesis needed)")
            continue

        if not s:
            continue

        shortfall = target - s.generated if s.generated < target else 0

        print(f"\n  {class_name}:")
        print(f"    Pure candidates: {s.candidates_found}")
        print(f"    Max possible (×{MAX_AUGMENTATIONS_PER_ORIGINAL}): {s.max_possible}")
        print(f"    Target: {target}")
        print(f"    Generated: {s.generated}")
        if shortfall > 0:
            print(f"    ⚠ Shortfall: {shortfall}")
        print(f"    Discarded (pHash): {s.discarded_phash}")
        print(f"    Discarded (pixel diff): {s.discarded_pixdiff}")

    total_new_glomeruli = sum(s.generated for s in stats_b.values())
    total_new_pairs = n_general + total_new_glomeruli

    print(f"\nTotal new pairs: {total_new_pairs}")
    print(f"Final dataset size: {n_base + total_new_pairs} pairs")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--tiles-dir',
        type=Path,
        default=Path('Salidas/Tiles_UNet'),
        help='Path to Tiles_UNet directory',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('Salidas/dataset_aug'),
        help='Path to output dataset_aug directory',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without writing files',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Delete and recreate output directory',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging',
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    check_dependencies()

    if not args.tiles_dir.exists():
        logger.error(f"Tiles directory not found: {args.tiles_dir}")
        sys.exit(1)

    output_root = args.output_dir.absolute()

    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be written")

    logger.info("Scanning dataset...")
    catalog_a, catalog_b = scan_dataset(args.tiles_dir)
    logger.info(f"Found {len(catalog_a)} annotated tiles")
    logger.info(f"Found {len(catalog_b['Esclerosado'])} pure Esclerosado tiles")
    logger.info(f"Found {len(catalog_b['Excluido'])} pure Excluido tiles")

    logger.info("Counting glomeruli by class...")
    glomeruli_original = count_glomeruli_by_class(args.tiles_dir)
    logger.info(f"Original distribution: {glomeruli_original}")

    synthetic_targets, glomeruli_after_a = calculate_synthetic_targets(
        glomeruli_original,
        TARGET_ESCLEROSADO,
        TARGET_EXCLUIDO,
        n_deterministic_transforms=len(DETERMINISTIC_TRANSFORMS),
    )
    logger.info(f"Synthetic targets needed: {synthetic_targets}")
    logger.info(f"Glomeruli after Component A: {glomeruli_after_a}")

    n_base = copy_base_dataset(args.tiles_dir, output_root, overwrite=args.overwrite)

    n_general, tile_log, augmented_a = run_general_augmentation(catalog_a, output_root, dry_run=args.dry_run)

    if not args.dry_run:
        log_path = output_root / 'tile_originals.json'
        with open(log_path, 'w') as f:
            json.dump({'tiles': tile_log}, f, indent=2)
        logger.info(f"Saved tile log: {log_path}")

    pipeline = build_augmentation_pipeline()
    stats_b, augmented_b = run_synthetic_generation(
        catalog_b,
        output_root,
        targets=synthetic_targets,
        pipeline=pipeline,
        max_aug=MAX_AUGMENTATIONS_PER_ORIGINAL,
        hash_threshold=PHASH_THRESHOLD,
        pixel_threshold=PIXEL_DIFF_THRESHOLD,
        dry_run=args.dry_run,
    )

    all_augmented = {}
    for biopsia, tiles in augmented_a.items():
        all_augmented.setdefault(biopsia, []).extend(tiles)
    for biopsia, tiles in augmented_b.items():
        all_augmented.setdefault(biopsia, []).extend(tiles)

    if not args.dry_run and all_augmented:
        update_annotations_json(output_root, all_augmented)

    glomeruli_final = dict(glomeruli_after_a)
    for class_name in ['Esclerosado', 'Excluido']:
        s = stats_b.get(class_name)
        if s:
            glomeruli_final[class_name] += s.generated

    print_report(
        n_base,
        n_general,
        glomeruli_original,
        glomeruli_after_a,
        synthetic_targets,
        glomeruli_final,
        stats_b,
        TARGET_ESCLEROSADO,
        TARGET_EXCLUIDO,
    )


if __name__ == '__main__':
    main()
