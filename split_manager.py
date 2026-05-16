"""
Shared biopsy split manager — ensures both U-Net and classification models
train/val/test on identical biopsy sets to avoid data leakage and enable fair
cross-model comparison.

Usage:
    python split_manager.py

Outputs:
    Salidas/biopsy_split.json — canonical split record
"""

import json
import logging
from pathlib import Path
from random import Random
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def _collect_image_tiles(images_dir: str) -> List[Path]:
    """Collect all PNG tile paths from a tiles directory."""
    images_dir = Path(images_dir)
    image_paths = sorted(images_dir.glob('*/images/*.png'))
    if not image_paths:
        image_paths = sorted(
            p for p in images_dir.rglob('*.png')
            if 'masks' not in p.relative_to(images_dir).parts
            and not p.stem.endswith('_mask')
        )
    return image_paths


def _slide_name_from_image_path(image_path: Path, images_dir: Path) -> str:
    """Infer biopsy/slide name from tile path."""
    try:
        rel = image_path.relative_to(images_dir)
        if len(rel.parts) >= 3 and rel.parts[1] == 'images':
            return rel.parts[0]
    except ValueError:
        pass

    if image_path.parent.name == 'images' and image_path.parent.parent.name:
        return image_path.parent.parent.name
    return image_path.parent.name or image_path.stem


def _group_images_by_biopsy(image_paths: List[Path], images_dir: str) -> Dict[str, List[Path]]:
    """Group tile paths by biopsy name."""
    images_dir = Path(images_dir)
    biopsias_dict = {}
    for img_path in image_paths:
        biopsia = _slide_name_from_image_path(img_path, images_dir)
        biopsias_dict.setdefault(biopsia, []).append(img_path)
    return biopsias_dict


def _safe_train_val_test_split(
    biopsias_list: List[str],
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split biopsias into train/val/test without crashing on tiny datasets."""
    test_size = 1.0 - train_size - val_size
    assert test_size >= 0, "train_size + val_size must be <= 1.0"

    biopsias_list = list(biopsias_list)
    if not biopsias_list:
        return [], [], []

    if len(biopsias_list) == 1:
        return biopsias_list, [], []

    if len(biopsias_list) == 2:
        rng = Random(seed)
        shuffled = biopsias_list[:]
        rng.shuffle(shuffled)
        return [shuffled[0]], [shuffled[1]], []

    if test_size > 0:
        train_val_biopsias, test_biopsias = train_test_split(
            biopsias_list,
            test_size=test_size,
            random_state=seed,
        )
    else:
        train_val_biopsias = biopsias_list
        test_biopsias = []

    if val_size > 0 and len(train_val_biopsias) > 1:
        val_fraction = val_size / (train_size + val_size)
        train_biopsias, val_biopsias = train_test_split(
            train_val_biopsias,
            test_size=val_fraction,
            random_state=seed + 1,
        )
    else:
        train_biopsias = train_val_biopsias
        val_biopsias = []

    return train_biopsias, val_biopsias, test_biopsias


def discover_shared_biopsies(tiles_dir: str, crops_dir: str) -> List[str]:
    """
    Find biopsies present in BOTH tiles and crops directories.
    Returns sorted list of shared biopsy names.
    """
    tiles_dir = Path(tiles_dir)
    crops_dir = Path(crops_dir)

    if tiles_dir.exists():
        tiles_biopsies = set(p.name for p in tiles_dir.iterdir() if p.is_dir() and (p / 'images').exists())
    else:
        tiles_biopsies = set()

    if crops_dir.exists():
        crops_biopsies = set(p.name for p in crops_dir.iterdir() if p.is_dir() and (p / 'images').exists())
    else:
        crops_biopsies = set()

    shared = sorted(tiles_biopsies & crops_biopsies)
    return shared


def generate_shared_split(
    tiles_dir: str = 'Salidas/Tiles_UNet',
    crops_dir: str = 'Salidas/Clasificador/crops',
    output_json: str = 'Salidas/biopsy_split.json',
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Generate and save a canonical split from shared biopsies across both datasets.

    Args:
        tiles_dir: Path to U-Net tiles directory
        crops_dir: Path to classification crops directory
        output_json: Where to save the split JSON
        train_size: Training set fraction
        val_size: Validation set fraction
        seed: Random seed for reproducibility

    Returns:
        Dict with keys 'train', 'val', 'test' (lists of biopsy names)
    """
    logger.info(f"Discovering shared biopsies between {tiles_dir} and {crops_dir}...")
    shared_biopsies = discover_shared_biopsies(tiles_dir, crops_dir)

    if not shared_biopsies:
        logger.error("No shared biopsies found between tiles and crops directories!")
        return {"train": [], "val": [], "test": []}

    logger.info(f"Found {len(shared_biopsies)} shared biopsies: {shared_biopsies}")

    train, val, test = _safe_train_val_test_split(
        shared_biopsies,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
    )

    split_dict = {
        "generated_at": str(Path(output_json).resolve()),
        "seed": seed,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": 1.0 - train_size - val_size,
        "total_biopsies": len(shared_biopsies),
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(split_dict, f, indent=2)

    logger.info(f"Split saved to {output_path}")
    logger.info(f"  Train: {len(train)} biopsies")
    logger.info(f"  Val:   {len(val)} biopsies")
    logger.info(f"  Test:  {len(test)} biopsies")

    return split_dict


def load_split(split_json: str = 'Salidas/biopsy_split.json') -> Dict:
    """Load a previously saved split from JSON."""
    split_path = Path(split_json)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_json}")

    with open(split_path, 'r') as f:
        return json.load(f)


def load_shared_split_for_dataset(
    split_json: str = 'Salidas/biopsy_split.json',
    dataset_dir: str = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load canonical split and filter to biopsies available in the given dataset.

    Args:
        split_json: Path to the canonical split JSON
        dataset_dir: Directory to scan for biopsies (e.g., 'Salidas/Tiles_UNet').
                    If None, returns split as-is (no filtering).

    Returns:
        Tuple of (train_biopsies, val_biopsies, test_biopsies)
    """
    split_dict = load_split(split_json)
    train = split_dict.get('train', [])
    val = split_dict.get('val', [])
    test = split_dict.get('test', [])

    if dataset_dir is None:
        return train, val, test

    # Filter to available biopsies
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        logger.warning(f"Dataset directory not found: {dataset_dir}, returning unfiltered split")
        return train, val, test

    available_biopsies = set(p.name for p in dataset_dir.iterdir() if p.is_dir())

    train_filtered = [b for b in train if b in available_biopsies]
    val_filtered = [b for b in val if b in available_biopsies]
    test_filtered = [b for b in test if b in available_biopsies]

    logger.info(f"Filtered split from {dataset_dir.name}:")
    logger.info(f"  Train: {len(train)} → {len(train_filtered)}")
    logger.info(f"  Val:   {len(val)} → {len(val_filtered)}")
    logger.info(f"  Test:  {len(test)} → {len(test_filtered)}")

    return train_filtered, val_filtered, test_filtered


if __name__ == '__main__':
    generate_shared_split()
