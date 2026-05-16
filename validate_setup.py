#!/usr/bin/env python
"""Validate that the shared split setup is working correctly."""

import json
import logging
from pathlib import Path
from split_manager import load_split, load_shared_split_for_dataset

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def validate_split_file():
    """Check that biopsy_split.json exists and is valid."""
    split_path = Path('Salidas/biopsy_split.json')

    if not split_path.exists():
        logger.error(f"❌ Split file not found: {split_path}")
        return False

    try:
        split = load_split(str(split_path))
    except Exception as e:
        logger.error(f"❌ Failed to load split: {e}")
        return False

    required_keys = {'train', 'val', 'test', 'seed', 'total_biopsies'}
    if not required_keys.issubset(split.keys()):
        logger.error(f"❌ Split missing keys: {required_keys - set(split.keys())}")
        return False

    train, val, test = split['train'], split['val'], split['test']

    # Check no overlap
    if set(train) & set(val):
        logger.error("❌ Train and Val have overlap!")
        return False
    if set(train) & set(test):
        logger.error("❌ Train and Test have overlap!")
        return False
    if set(val) & set(test):
        logger.error("❌ Val and Test have overlap!")
        return False

    logger.info("✅ Split file is valid")
    logger.info(f"   Train: {len(train)} biopsies")
    logger.info(f"   Val:   {len(val)} biopsies")
    logger.info(f"   Test:  {len(test)} biopsies")
    logger.info(f"   Total: {len(train) + len(val) + len(test)} biopsies")
    logger.info(f"   Seed:  {split['seed']}")

    return True


def validate_dataset_directories():
    """Check that dataset directories exist."""
    tiles_dir = Path('Salidas/Tiles_UNet')
    crops_dir = Path('Salidas/Clasificador/crops')

    tiles_ok = tiles_dir.exists()
    crops_ok = crops_dir.exists()

    if not tiles_ok:
        logger.warning(f"⚠️  Tiles directory not found: {tiles_dir}")
    else:
        tiles_count = len(list(tiles_dir.iterdir()))
        logger.info(f"✅ Tiles directory exists ({tiles_count} subdirs)")

    if not crops_ok:
        logger.warning(f"⚠️  Crops directory not found: {crops_dir}")
    else:
        crops_count = len(list(crops_dir.iterdir()))
        logger.info(f"✅ Crops directory exists ({crops_count} subdirs)")

    return tiles_ok and crops_ok


def validate_split_loading():
    """Test load_shared_split_for_dataset function."""
    logger.info("\n--- Testing Split Loading ---")

    try:
        train, val, test = load_shared_split_for_dataset(
            'Salidas/biopsy_split.json',
            'Salidas/Tiles_UNet'
        )
        logger.info(f"✅ Successfully loaded split for Tiles_UNet")
        logger.info(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    except Exception as e:
        logger.error(f"❌ Failed to load split for Tiles_UNet: {e}")
        return False

    try:
        train, val, test = load_shared_split_for_dataset(
            'Salidas/biopsy_split.json',
            'Salidas/Clasificador/crops'
        )
        logger.info(f"✅ Successfully loaded split for Clasificador/crops")
        logger.info(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    except Exception as e:
        logger.error(f"❌ Failed to load split for Clasificador/crops: {e}")
        return False

    return True


def validate_masks_and_indices():
    """Check if masks and glomerulus indices exist (optional)."""
    logger.info("\n--- Checking Masks & Indices (Optional) ---")

    imagen_dir = Path('Salidas/Imagen')
    if not imagen_dir.exists():
        logger.warning("⚠️  Salidas/Imagen directory not found (run tiles.py to generate)")
        return True

    mask_count = len(list(imagen_dir.glob('*/masks/*.png')))
    index_count = len(list(imagen_dir.glob('*/glomerulus_index.json')))

    if mask_count > 0:
        logger.info(f"✅ Found {mask_count} mask files")
    else:
        logger.warning("⚠️  No mask files found (run tiles.py with annotations_dir)")

    if index_count > 0:
        logger.info(f"✅ Found {index_count} glomerulus index files")
    else:
        logger.warning("⚠️  No index files found (run tiles.py with annotations_dir)")

    return True


def main():
    """Run all validations."""
    logger.info("=" * 60)
    logger.info("VALIDATING SHARED SPLIT SETUP")
    logger.info("=" * 60)

    checks = [
        ("Split File", validate_split_file),
        ("Dataset Directories", validate_dataset_directories),
        ("Split Loading", validate_split_loading),
        ("Masks & Indices", validate_masks_and_indices),
    ]

    results = []
    for name, check_fn in checks:
        logger.info(f"\n--- {name} ---")
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ Exception in {name}: {e}")
            results.append((name, False))

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for name, result in results:
        status = "✅" if result else "❌"
        logger.info(f"{status} {name}")

    all_ok = all(r[1] for r in results)

    if all_ok:
        logger.info("\n✅ All checks passed! Ready to train.")
    else:
        logger.warning("\n⚠️  Some checks failed. See above for details.")

    return all_ok


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
