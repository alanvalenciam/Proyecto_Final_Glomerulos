"""Generate tiled training data for UNet from WSI TIFF + GeoJSON annotations."""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from PIL import Image
import psutil

try:
    import openslide
except ImportError:
    openslide = None

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from skimage import draw
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from shapely.geometry import box, Polygon
except ImportError:
    raise ImportError("shapely is required: pip install shapely")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

# Configuration
TILE_SIZE = 1024
ZOOM_SCALE = 0.5
STRIDE = 512
MIN_COVERAGE_PCT = 60.0
OUTPUT_FORMAT = 'png'
COMPRESSION = 0

LARGE_FILE_THRESHOLD = 300 * 1024 * 1024  # 300 MB

# Explicit class priority for mask overlaps (higher = higher priority)
# When polygons overlap, the class with the highest priority wins
CLASS_PRIORITY = {
    0: 0,      # background (lowest priority)
    1: 10,     # No_Proliferativo
    2: 8,      # Proliferativo
    3: 5,      # Esclerosado
    4: 1,      # Excluido (lowest priority despite highest class_id)
}

# Class mapping — CANONICAL and FIXED (deterministic)
CANONICAL_CLASS_MAP = {
    'background':       0,
    'No_Proliferativo': 1,
    'Proliferativo':    2,
    'Esclerosado':      3,
    'Excluido':         4,
}

CLASS_NAME_NORMALIZE = {
    # No_Proliferativo (keys normalized: lowercase, no spaces)
    'noprolif': 'No_Proliferativo',
    'noproliferativo': 'No_Proliferativo',
    'no_proliferativo': 'No_Proliferativo',
    # Proliferativo
    'proliferativo': 'Proliferativo',
    'prolif': 'Proliferativo',
    # Esclerosado
    'esclerosado': 'Esclerosado',
    'sclerosed': 'Esclerosado',
    'sclerotic': 'Esclerosado',
    # Excluido
    'exclude': 'Excluido',
    'excluido': 'Excluido',
    'excluded': 'Excluido',
}

# Tissue filtering parameters (from tiles.py)
OTSU_MIN_RATIO = 0.01     # 1% minimum tissue in Otsu mask
BG_THRESHOLD = 15.0       # minimum std in grayscale
DOWNSAMPLE_FACTOR = 20    # for Otsu mask generation

# ============================================================================
# Tissue filtering functions (from tiles.py)
# ============================================================================

def generate_otsu_mask(img_or_slide: object, downsample_factor: int = 20) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Generate binary tissue mask via Otsu thresholding on HSV saturation."""
    if not OPENCV_AVAILABLE:
        logger.warning("OpenCV not available, skipping Otsu mask")
        return None, None

    if hasattr(img_or_slide, 'get_thumbnail'):  # OpenSlide — memory-efficient
        w, h = img_or_slide.level_dimensions[0]
        thumb = img_or_slide.get_thumbnail((w // downsample_factor, h // downsample_factor))
    else:  # PIL Image — three-stage aggressive downsampling to avoid full load
        w, h = img_or_slide.size
        target_w = w // downsample_factor
        target_h = h // downsample_factor

        try:
            if hasattr(img_or_slide, 'draft'):
                img_or_slide.draft('RGB', (target_w, target_h))
            if w > 20000 or h > 20000:
                stage1 = w // 50
                stage2 = w // 5
                intermediate1 = img_or_slide.resize((stage1, h // 50), Image.Resampling.NEAREST)
                intermediate2 = intermediate1.resize((stage2, h // 5), Image.Resampling.BILINEAR)
                thumb = intermediate2.resize((target_w, target_h), Image.Resampling.LANCZOS)
            elif w > 10000 or h > 10000:
                scale = max(1, downsample_factor // 2)
                intermediate = img_or_slide.resize((w // scale, h // scale), Image.Resampling.NEAREST)
                thumb = intermediate.resize((target_w, target_h), Image.Resampling.LANCZOS)
            else:
                thumb = img_or_slide.resize((target_w, target_h), Image.Resampling.LANCZOS)
        except MemoryError:
            logger.warning(f"MemoryError during Otsu resize, using extreme downsampling")
            extreme_scale = max(1, downsample_factor * 5)
            thumb = img_or_slide.resize((w // extreme_scale, h // extreme_scale), Image.Resampling.NEAREST)

    thumb_cv = cv2.cvtColor(np.array(thumb.convert('RGB')), cv2.COLOR_RGB2HSV)
    saturation = cv2.medianBlur(thumb_cv[:, :, 1], 7)
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, downsample_factor

def has_tissue_in_mask(mask: Optional[np.ndarray], x: int, y: int, tile_size: int, downsample_factor: int, min_ratio: float = 0.01) -> bool:
    """Fast pre-check if tile region contains tissue via Otsu mask."""
    if mask is None:
        return True

    mx = int(x / downsample_factor)
    my = int(y / downsample_factor)
    ms = int(tile_size / downsample_factor)

    patch = mask[my:my + ms, mx:mx + ms]
    if patch.size == 0:
        return False

    tissue_pixels = np.count_nonzero(patch)
    tissue_ratio = tissue_pixels / patch.size
    return tissue_ratio >= min_ratio

def is_mostly_background(tile: Image.Image, min_std: float = 15.0) -> bool:
    """Return True if tile lacks tissue (should be skipped)."""
    gray = np.array(tile.convert('L'))
    return float(np.std(gray)) < min_std

# ============================================================================
# Utility functions
# ============================================================================

def normalize_class_name(raw: str) -> str:
    """Normalize raw class name from GeoJSON to canonical form."""
    key = raw.strip().lower().replace(' ', '')
    normalized = CLASS_NAME_NORMALIZE.get(key)
    if normalized:
        return normalized
    # If not found, try direct canonical match
    for canonical in CANONICAL_CLASS_MAP.keys():
        if key == canonical.lower().replace('_', ''):
            return canonical
    logger.warning(f"Unknown class name '{raw}', using as-is")
    return raw

def estimate_memory(tiff_path: str, expansion_factor: float = 5.0) -> float:
    """Estimate RAM needed to process a TIFF file (in GB)."""
    file_size_gb = os.path.getsize(tiff_path) / (1024**3)
    return file_size_gb * expansion_factor

def get_available_ram_gb() -> float:
    """Get available system RAM in GB."""
    return psutil.virtual_memory().available / (1024**3)

def detect_format(file_path: str) -> Optional[str]:
    """Detect image format: 'openslide', 'pil', or None."""
    ext = Path(file_path).suffix.lower()
    size = os.path.getsize(file_path)

    # For TIFF files, strongly prefer OpenSlide if available (WSI-efficient)
    if ext in {'.tif', '.tiff'}:
        if openslide and size > LARGE_FILE_THRESHOLD:
            return 'openslide'
        # PIL for smaller TIFFs only
        if size < LARGE_FILE_THRESHOLD:
            return 'pil'
        # Large TIFF without OpenSlide: warn and try PIL anyway (will be slow)
        if not openslide and size > LARGE_FILE_THRESHOLD:
            logger.warning(f"Large TIFF file ({size / 1e9:.1f} GB) without OpenSlide. Install 'openslide' for efficiency: pip install openslide-python")
            return 'pil'
        return 'pil'

    # SVS, NDPI, VMS: OpenSlide only
    if openslide and ext in {'.svs', '.ndpi', '.vms'}:
        return 'openslide'

    # Regular images
    if ext in {'.png', '.jpg', '.jpeg'}:
        return 'pil'

    return None

def load_slide(tiff_path: str) -> Tuple[object, int, int, str]:
    """Load slide and return (slide_handle, width, height, format_type)."""
    fmt = detect_format(tiff_path)
    if fmt == 'openslide':
        try:
            slide = openslide.open_slide(tiff_path)
            w, h = slide.dimensions
            return slide, w, h, 'openslide'
        except Exception as e:
            logger.warning(f"OpenSlide failed on {tiff_path}: {e}. Falling back to PIL.")

    # PIL fallback
    try:
        img = Image.open(tiff_path)
        w, h = img.size
        return img, w, h, 'pil'
    except Exception as e:
        logger.error(f"Failed to open {tiff_path}: {e}")
        raise

def extract_tile_pil(slide: Image.Image, x: int, y: int, size: int) -> Image.Image:
    """Extract a tile from a PIL image."""
    x_end = min(x + size, slide.width)
    y_end = min(y + size, slide.height)
    tile = slide.crop((x, y, x_end, y_end))

    if tile.size != (size, size):
        padded = Image.new('RGB', (size, size), (255, 255, 255))
        padded.paste(tile, (0, 0))
        tile = padded

    return tile

def extract_tile_openslide(slide: object, x: int, y: int, size: int, level: int = 0) -> Image.Image:
    """Extract a tile from an OpenSlide slide."""
    tile = slide.read_region((x, y), level, (size, size))
    tile = tile.convert('RGB')
    return tile

def load_geojson(geojson_path: str) -> Dict:
    """Load a GeoJSON file."""
    with open(geojson_path, 'r') as f:
        return json.load(f)

def find_tiff_geojson_pairs(input_dir: str) -> List[Dict]:
    """Find matched pairs of .tiff and .geojson files by stem."""
    input_path = Path(input_dir)
    tiffs = {f.stem: f for f in input_path.glob('*.tiff')} | \
            {f.stem: f for f in input_path.glob('*.tif')}
    geojsons = {f.stem: f for f in input_path.glob('*.geojson')}

    pairs = []
    for stem in sorted(tiffs.keys()):
        if stem in geojsons:
            pairs.append({
                'stem': stem,
                'image_path': str(tiffs[stem]),
                'geojson_path': str(geojsons[stem])
            })

    logger.info(f"Found {len(pairs)} TIFF+GeoJSON pairs in {input_dir}")
    return pairs

def get_canonical_class_map() -> Dict[str, int]:
    """Return the canonical class map (deterministic, no scanning needed)."""
    logger.info(f"Using canonical class map: {CANONICAL_CLASS_MAP}")
    return CANONICAL_CLASS_MAP

# ============================================================================
# Tiling and mask generation
# ============================================================================

def generate_grid_tiles(
    width: int,
    height: int,
    tile_size: int = TILE_SIZE,
    stride: int = STRIDE
) -> List[Tuple[int, int, int, int]]:
    """Generate grid tile bounding boxes (native coordinates)."""
    tiles = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tiles.append((x, y, x + tile_size, y + tile_size))

    # Right edge tiles
    x = width - tile_size
    if x >= 0:
        for y in range(0, height - tile_size + 1, stride):
            if (x, y, x + tile_size, y + tile_size) not in tiles:
                tiles.append((x, y, x + tile_size, y + tile_size))

    # Bottom edge tiles
    y = height - tile_size
    if y >= 0:
        for x in range(0, width - tile_size + 1, stride):
            if (x, y, x + tile_size, y + tile_size) not in tiles:
                tiles.append((x, y, x + tile_size, y + tile_size))

    # Bottom-right corner
    if x >= 0 and y >= 0:
        if (x, y, x + tile_size, y + tile_size) not in tiles:
            tiles.append((x, y, x + tile_size, y + tile_size))

    return tiles

def polygon_to_shapely(polygon_coords: List[List[float]]) -> Polygon:
    """Convert polygon coordinate list to Shapely Polygon."""
    return Polygon(polygon_coords)

def compute_polygon_centroid(polygon_coords: List[List[float]]) -> Tuple[float, float]:
    """Compute centroid of a polygon."""
    poly = Polygon(polygon_coords)
    centroid = poly.centroid
    return centroid.x, centroid.y

def rasterize_polygon(
    polygon_coords: List[List[float]],
    tile_box: Tuple[int, int, int, int],
    tile_size: int = TILE_SIZE
) -> np.ndarray:
    """Rasterize a polygon to a numpy array within a tile's bounding box."""
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)

    try:
        polygon = polygon_to_shapely(polygon_coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        tile_poly = box(tile_box[0], tile_box[1], tile_box[2], tile_box[3])
        intersection = polygon.intersection(tile_poly)

        if intersection.is_empty or intersection.area == 0:
            return mask

        from shapely.geometry import GeometryCollection, MultiPolygon, LineString
        if isinstance(intersection, GeometryCollection):
            geoms = list(intersection.geoms)
        elif isinstance(intersection, MultiPolygon):
            geoms = list(intersection.geoms)
        else:
            geoms = [intersection]

        for geom in geoms:
            if geom.is_empty or isinstance(geom, LineString):
                continue
            if hasattr(geom, 'exterior'):
                coords = list(geom.exterior.coords)
                tile_coords = [
                    (c[0] - tile_box[0], c[1] - tile_box[1])
                    for c in coords
                ]
                if len(tile_coords) > 2:
                    rr, cc = _bresenham_polygon(tile_coords, tile_size)
                    if len(rr) > 0:
                        mask[rr, cc] = 255
    except Exception as e:
        logger.warning(f"Failed to rasterize polygon: {e}")

    return mask

def _bresenham_polygon(coords: List[Tuple[int, int]], size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize polygon interior using scan-line fill algorithm."""
    if SKIMAGE_AVAILABLE:
        try:
            rr, cc = draw.polygon([c[1] for c in coords], [c[0] for c in coords], shape=(size, size))
            return rr, cc
        except Exception as e:
            logger.debug(f"skimage.draw.polygon failed: {e}")
            pass

    if len(coords) < 3:
        logger.debug(f"Polygon has {len(coords)} coords, need >= 3")
        return np.array([], dtype=int), np.array([], dtype=int)

    rows = [int(round(c[1])) for c in coords]
    cols = [int(round(c[0])) for c in coords]

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    min_row = max(0, min_row)
    max_row = min(size - 1, max_row)
    min_col = max(0, min_col)
    max_col = min(size - 1, max_col)

    filled_rr, filled_cc = [], []

    for row in range(min_row, max_row + 1):
        intersections = []
        for i in range(len(coords)):
            y1 = rows[i]
            y2 = rows[(i + 1) % len(coords)]
            x1 = cols[i]
            x2 = cols[(i + 1) % len(coords)]

            if (y1 <= row <= y2) or (y2 <= row <= y1):
                if y1 != y2:
                    x_intersect = x1 + (row - y1) * (x2 - x1) / (y2 - y1)
                    intersections.append(x_intersect)
                elif y1 == row:
                    intersections.append(x1)
                    intersections.append(x2)

        if len(intersections) >= 2:
            intersections.sort()
            for j in range(0, len(intersections), 2):
                if j + 1 < len(intersections):
                    col_start = max(0, int(intersections[j]))
                    col_end = min(size - 1, int(intersections[j + 1]))
                    for col in range(col_start, col_end + 1):
                        filled_rr.append(row)
                        filled_cc.append(col)

    return np.array(filled_rr, dtype=int), np.array(filled_cc, dtype=int)

def compute_coverage(
    polygon_coords: List[List[float]],
    tile_box: Tuple[int, int, int, int]
) -> float:
    """Compute percentage of polygon area that overlaps with tile_box."""
    try:
        polygon = polygon_to_shapely(polygon_coords)
        if polygon.area == 0:
            return 0.0

        tile_poly = box(tile_box[0], tile_box[1], tile_box[2], tile_box[3])
        intersection = polygon.intersection(tile_poly)

        return (intersection.area / polygon.area) * 100.0
    except Exception as e:
        logger.warning(f"Failed to compute coverage: {e}")
        return 0.0

def combine_masks_with_priority(
    current_mask: np.ndarray,
    new_mask: np.ndarray,
    new_class_id: int
) -> np.ndarray:
    """Combine two class masks using explicit priority-based logic."""
    result = current_mask.copy()

    # Find pixels where new_mask has annotation (non-zero)
    new_pixels = new_mask > 0
    new_priority = CLASS_PRIORITY.get(new_class_id, 0)

    for idx_flat in np.where(new_pixels.ravel())[0]:
        row, col = np.unravel_index(idx_flat, new_pixels.shape)
        current_class_id = result[row, col]
        current_priority = CLASS_PRIORITY.get(current_class_id, 0)

        # New class wins if it has higher priority
        if new_priority > current_priority:
            result[row, col] = 255

    return result

# ============================================================================
# Primary/Secondary tile assignment
# ============================================================================

def compute_primary_secondary(
    glomeruli: List[Dict],
    grid_tiles: List[Tuple[int, int, int, int]],
    coverage_threshold: float = MIN_COVERAGE_PCT
) -> Dict[int, Dict]:
    """For each glomerulus, determine primary and secondary tiles."""
    assignment = {}

    for glom_id, glom_data in enumerate(glomeruli):
        polygon = glom_data['coordinates']

        # Compute coverage in all grid tiles
        coverages = []
        for tile_idx, tile_box in enumerate(grid_tiles):
            cov = compute_coverage(polygon, tile_box)
            if cov > 0:
                coverages.append((tile_idx, cov))

        # Sort by coverage descending
        coverages.sort(key=lambda x: x[1], reverse=True)

        if coverages and coverages[0][1] >= coverage_threshold:
            # Primary is best grid tile
            primary_idx, primary_cov = coverages[0]
            secondary_idx = coverages[1][0] if len(coverages) > 1 else None
            secondary_cov = coverages[1][1] if len(coverages) > 1 else 0.0

            assignment[glom_id] = {
                'primary_tile_idx': primary_idx,
                'secondary_tile_idx': secondary_idx,
                'primary_coverage': primary_cov,
                'secondary_coverage': secondary_cov,
                'primary_is_centered': False,
                'needs_centered_tile': False
            }
        else:
            # Need a centered tile
            secondary_idx = coverages[0][0] if coverages else None
            secondary_cov = coverages[0][1] if coverages else 0.0

            assignment[glom_id] = {
                'primary_tile_idx': None,  # Will trigger centered tile creation
                'secondary_tile_idx': secondary_idx,
                'primary_coverage': 0.0,  # PLACEHOLDER — will be updated after centered tile created
                'secondary_coverage': secondary_cov,
                'primary_is_centered': True,
                'needs_centered_tile': True
            }

    return assignment

def update_coverage_for_centered_tile(
    glom_id: int,
    glom_coords: List[List[float]],
    tile_box: Tuple[int, int, int, int],
    assignment: Dict[int, Dict]
) -> None:
    """Update primary_coverage for a glomerulus that needed a centered tile."""
    if glom_id in assignment and assignment[glom_id]['primary_is_centered']:
        real_coverage = compute_coverage(glom_coords, tile_box)
        assignment[glom_id]['primary_coverage'] = real_coverage

# ============================================================================
# Tile I/O
# ============================================================================

def save_tile_image(image: Image.Image, output_path: str):
    """Save a tile image as PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, OUTPUT_FORMAT.upper(), compress_level=COMPRESSION)

def save_tile_mask(mask: np.ndarray, output_path: str):
    """Save a tile mask as PNG uint8."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray(mask, mode='L')
    mask_img.save(output_path, 'PNG', compress_level=COMPRESSION)

# ============================================================================
# Slide processing (main work function)
# ============================================================================

def process_slide_pair(
    pair: Dict,
    output_dir: str,
    class_map: Dict[str, int],
    tile_size: int = TILE_SIZE,
    zoom_scale: float = ZOOM_SCALE,
    stride: int = STRIDE
) -> Dict:
    """Process a single TIFF+GeoJSON pair and generate tiles, masks, and metadata."""
    stem = pair['stem']
    image_path = pair['image_path']
    geojson_path = pair['geojson_path']

    try:
        logger.info(f"[{stem}] Loading TIFF and GeoJSON...")
        slide, width, height, fmt = load_slide(image_path)
        geojson_data = load_geojson(geojson_path)

        # Generate Otsu tissue mask for filtering
        logger.info(f"[{stem}] Computing Otsu tissue mask...")
        otsu_mask, ds_factor = generate_otsu_mask(slide, downsample_factor=DOWNSAMPLE_FACTOR)
        native_tile_size = int(tile_size / zoom_scale)

        # Extract glomeruli from GeoJSON with class normalization
        glomeruli = []
        for feat_idx, feature in enumerate(geojson_data.get('features', [])):
            geom = feature.get('geometry', {})
            if geom.get('type') != 'Polygon':
                continue

            props = feature.get('properties', {})
            classification = props.get('classification', {})

            if isinstance(classification, dict):
                raw_class_name = classification.get('name', 'unknown')
            else:
                raw_class_name = str(classification) if classification else 'unknown'

            # Normalize class name
            normalized_class_name = normalize_class_name(raw_class_name)
            class_id = class_map.get(normalized_class_name, class_map.get('background', 0))

            coords = geom.get('coordinates', [[]])[0]
            if len(coords) < 3:
                logger.warning(f"[{stem}] Feature {feat_idx} has < 3 coordinates, skipping")
                continue

            glomeruli.append({
                'id': len(glomeruli),
                'feature_index': feat_idx,
                'raw_class_name': raw_class_name,
                'class_name': normalized_class_name,
                'class_id': class_id,
                'coordinates': coords
            })

        logger.info(f"[{stem}] Found {len(glomeruli)} glomeruli")

        # Generate grid tiles
        logger.info(f"[{stem}] Generating grid tiles (size={tile_size}, stride={stride})...")
        grid_tiles = generate_grid_tiles(width, height, tile_size, stride)
        logger.info(f"[{stem}] Generated {len(grid_tiles)} grid tiles")

        # Compute primary/secondary assignments
        logger.info(f"[{stem}] Computing primary/secondary tile assignments...")
        assignments = compute_primary_secondary(glomeruli, grid_tiles)

        # Determine which centered tiles are needed
        centered_glom_ids = [gid for gid, asg in assignments.items() if asg['needs_centered_tile']]
        logger.info(f"[{stem}] {len(centered_glom_ids)} glomeruli need centered tiles")

        # Create output directory
        slide_output_dir = Path(output_dir) / stem
        images_dir = slide_output_dir / 'images'
        masks_dir = slide_output_dir / 'masks'
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Build tile registry for metadata
        tile_registry = []

        # Process grid tiles with filtering
        logger.info(f"[{stem}] Extracting {len(grid_tiles)} grid tiles with tissue filtering...")
        tiles_saved = 0
        tiles_skipped = 0

        for tile_idx, (x, y, x_end, y_end) in enumerate(grid_tiles):
            # New tile naming: native coordinates with 5-digit padding
            tile_name = f"{stem}_tile_x{x:05d}_y{y:05d}_endx{x_end:05d}_endy{y_end:05d}"

            # Extract tile image
            if fmt == 'openslide':
                tile_img = extract_tile_openslide(slide, x, y, tile_size)
            else:
                tile_img = extract_tile_pil(slide, x, y, tile_size)

            mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            tile_glomeruli = []

            for glom_id, glom_data in enumerate(glomeruli):
                cov = compute_coverage(glom_data['coordinates'], (x, y, x_end, y_end))
                if cov > 0:
                    # Rasterize polygon onto mask
                    glom_mask = rasterize_polygon(
                        glom_data['coordinates'],
                        (x, y, x_end, y_end),
                        tile_size
                    )
                    # Combine masks using explicit priority (not implicit np.maximum)
                    mask = combine_masks_with_priority(mask, glom_mask, glom_data['class_id'])

                    role = 'primary' if assignments[glom_id]['primary_tile_idx'] == tile_idx else 'secondary'
                    tile_glomeruli.append({
                        'id': glom_id,
                        'class': glom_data['class_name'],
                        'coverage_pct': cov,
                        'role': role
                    })

            # Filtering: tiles WITH annotations are ALWAYS saved (ground truth takes priority)
            # Tiles WITHOUT annotations are filtered for tissue
            has_annotations = len(tile_glomeruli) > 0

            if not has_annotations:
                # Apply tissue filters only to tiles without annotations
                if not has_tissue_in_mask(otsu_mask, x, y, native_tile_size, ds_factor, OTSU_MIN_RATIO):
                    tiles_skipped += 1
                    continue
                if is_mostly_background(tile_img, BG_THRESHOLD):
                    tiles_skipped += 1
                    continue

            # Save tile and mask
            img_path = images_dir / f"{tile_name}.{OUTPUT_FORMAT}"
            mask_path = masks_dir / f"{tile_name}_mask.{OUTPUT_FORMAT}"
            save_tile_image(tile_img, str(img_path))
            save_tile_mask(mask, str(mask_path))
            tiles_saved += 1

            # Record in registry (always for annotated tiles, or for saved tissue tiles)
            tile_registry.append({
                'type': 'grid',
                'image': str(img_path.relative_to(slide_output_dir)),
                'mask': str(mask_path.relative_to(slide_output_dir)),
                'origin_native': [x, y],
                'tile_index': tile_idx,
                'has_annotations': has_annotations,
                'glomeruli': tile_glomeruli
            })

        logger.info(f"[{stem}] Grid tiles: {tiles_saved} saved, {tiles_skipped} skipped (background)")

        # Process centered tiles
        logger.info(f"[{stem}] Extracting {len(centered_glom_ids)} centered tiles...")
        for glom_id in centered_glom_ids:
            glom_data = glomeruli[glom_id]
            cx, cy = compute_polygon_centroid(glom_data['coordinates'])

            # Tile box centered on glomerulus
            x = int(max(0, cx - tile_size / 2))
            y = int(max(0, cy - tile_size / 2))
            x = min(x, width - tile_size)
            y = min(y, height - tile_size)
            x_end, y_end = x + tile_size, y + tile_size

            # Calculate real coverage in the centered tile (not placeholder 0.0)
            update_coverage_for_centered_tile(glom_id, glom_data['coordinates'], (x, y, x_end, y_end), assignments)

            tile_name = f"{stem}_centered_g{glom_id:04d}"

            # Extract tile image
            if fmt == 'openslide':
                tile_img = extract_tile_openslide(slide, x, y, tile_size)
            else:
                tile_img = extract_tile_pil(slide, x, y, tile_size)

            mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            tile_glomeruli = []

            for gid, gdata in enumerate(glomeruli):
                cov = compute_coverage(gdata['coordinates'], (x, y, x_end, y_end))
                if cov > 0:
                    glom_mask = rasterize_polygon(
                        gdata['coordinates'],
                        (x, y, x_end, y_end),
                        tile_size
                    )
                    # Combine masks using explicit priority (not implicit np.maximum)
                    mask = combine_masks_with_priority(mask, glom_mask, gdata['class_id'])

                    role = 'primary' if gid == glom_id else 'secondary'
                    tile_glomeruli.append({
                        'id': gid,
                        'class': gdata['class_name'],
                        'coverage_pct': cov,
                        'role': role
                    })

            # Save
            img_path = images_dir / f"{tile_name}.{OUTPUT_FORMAT}"
            mask_path = masks_dir / f"{tile_name}_mask.{OUTPUT_FORMAT}"
            save_tile_image(tile_img, str(img_path))
            save_tile_mask(mask, str(mask_path))

            tile_registry.append({
                'type': 'centered',
                'image': str(img_path.relative_to(slide_output_dir)),
                'mask': str(mask_path.relative_to(slide_output_dir)),
                'origin_native': [x, y],
                'glomerulus_id': glom_id,
                'glomeruli': tile_glomeruli
            })

        # Build glomeruli coverage index
        glomeruli_coverage = {}
        for glom_id, asg in assignments.items():
            primary_tile_name = None
            secondary_tile_name = None

            if asg['primary_is_centered']:
                primary_tile_name = f"{stem}_centered_g{glom_id:04d}"
            else:
                tile_idx = asg['primary_tile_idx']
                x, y, x_end, y_end = grid_tiles[tile_idx]
                primary_tile_name = f"{stem}_tile_x{x:05d}_y{y:05d}_endx{x_end:05d}_endy{y_end:05d}"

            if asg['secondary_tile_idx'] is not None:
                tile_idx = asg['secondary_tile_idx']
                x, y, x_end, y_end = grid_tiles[tile_idx]
                secondary_tile_name = f"{stem}_tile_x{x:05d}_y{y:05d}_endx{x_end:05d}_endy{y_end:05d}"

            glomeruli_coverage[glom_id] = {
                'class': glomeruli[glom_id]['class_name'],
                'primary_tile': primary_tile_name,
                'secondary_tile': secondary_tile_name,
                'primary_coverage_pct': asg['primary_coverage'],
                'secondary_coverage_pct': asg['secondary_coverage']
            }

        # Save annotations.json
        annotations = {
            'slide': stem,
            'width_native': width,
            'height_native': height,
            'tile_size': tile_size,
            'zoom_scale': zoom_scale,
            'stride': stride,
            'class_map': CANONICAL_CLASS_MAP,
            'n_glomeruli': len(glomeruli),
            'n_grid_tiles': len(grid_tiles),
            'n_centered_tiles': len(centered_glom_ids),
            'otsu_applied': otsu_mask is not None,
            'glomeruli_coverage': glomeruli_coverage,
            'tiles': tile_registry
        }

        annotations_path = slide_output_dir / 'annotations.json'
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)

        logger.info(f"[{stem}] ✓ Complete: {len(grid_tiles)} grid + {len(centered_glom_ids)} centered = "
                   f"{len(grid_tiles) + len(centered_glom_ids)} total tiles")

        return {
            'status': 'success',
            'slide_name': stem,
            'message': f"Generated {len(grid_tiles) + len(centered_glom_ids)} tiles"
        }

    except Exception as e:
        logger.error(f"[{stem}] Error: {e}", exc_info=True)
        return {
            'status': 'error',
            'slide_name': stem,
            'message': str(e)
        }

# ============================================================================
# Dynamic scheduling
# ============================================================================

def dynamic_schedule(
    pairs: List[Dict],
    output_dir: str,
    class_map: Dict[str, int],
    ram_fraction: float = 0.9,
    tile_size: int = TILE_SIZE,
    max_workers: int = None
) -> List[Dict]:
    """Process pairs with dynamic worker admission based on available RAM."""
    queue = deque(sorted(pairs, key=lambda p: os.path.getsize(p['image_path']), reverse=False))
    active = {}  # future → (pair, mem_estimate)
    results = []

    executor = ProcessPoolExecutor(max_workers=max_workers or cpu_count())
    ram_budget_gb = get_available_ram_gb() * ram_fraction
    used_ram_gb = 0.0

    logger.info(f"RAM budget: {ram_budget_gb:.1f} GB ({ram_fraction*100:.0f}% of {get_available_ram_gb():.1f} GB)")

    def try_submit():
        nonlocal used_ram_gb
        while queue:
            pair = queue[0]
            mem_estimate = estimate_memory(pair['image_path'])

            if used_ram_gb + mem_estimate <= ram_budget_gb:
                queue.popleft()
                logger.info(f"Submitting {pair['stem']} ({mem_estimate:.1f} GB, total: {used_ram_gb + mem_estimate:.1f} GB)")

                future = executor.submit(
                    process_slide_pair,
                    pair,
                    output_dir,
                    class_map,
                    tile_size
                )
                active[future] = (pair, mem_estimate)
                used_ram_gb += mem_estimate
            else:
                logger.info(f"Waiting for slot: {pair['stem']} needs {mem_estimate:.1f} GB, "
                           f"only {ram_budget_gb - used_ram_gb:.1f} GB available")
                break

    try_submit()

    while active:
        done, _ = wait(active, return_when=FIRST_COMPLETED)
        for future in done:
            pair, mem = active.pop(future)
            try:
                result = future.result()
                results.append(result)
                logger.info(f"{pair['stem']}: {result['message']}")
            except Exception as e:
                logger.error(f"{pair['stem']}: {e}")
                results.append({
                    'status': 'error',
                    'slide_name': pair['stem'],
                    'message': str(e)
                })

            used_ram_gb -= mem
            logger.info(f"Freed {mem:.1f} GB, now using {used_ram_gb:.1f} GB")

        try_submit()

    executor.shutdown(wait=True)
    logger.info(f"All {len(pairs)} slides processed")
    return results

# ============================================================================
# CLI
# ============================================================================

def slide_already_processed(stem: str, output_dir: str) -> bool:
    """Check if a slide's output directory exists with tiles."""
    slide_dir = Path(output_dir) / stem
    images_dir = slide_dir / 'images'
    return images_dir.exists() and list(images_dir.glob('*.png'))

def main():
    parser = argparse.ArgumentParser(
        description='Generate tiled training data for UNet from WSI + GeoJSON annotations'
    )
    parser.add_argument(
        '--input', '-i',
        default='Entradas',
        help='Input directory with .tiff and .geojson pairs'
    )
    parser.add_argument(
        '--output', '-o',
        default='Salidas/Tiles_UNet',
        help='Output directory'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=TILE_SIZE,
        help='Tile size in pixels (at zoom_scale)'
    )
    parser.add_argument(
        '--zoom-scale',
        type=float,
        default=ZOOM_SCALE,
        help='Zoom scale (1 tile pixel = 1/zoom_scale native pixels)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=STRIDE,
        help='Stride between tiles (default = tile_size - overlap for 50% overlap)'
    )
    parser.add_argument(
        '--ram-fraction',
        type=float,
        default=0.9,
        help='Fraction of available RAM to use (0-1)'
    )
    parser.add_argument(
        '--max-slides',
        type=int,
        default=None,
        help='Limit number of slides to process (for testing)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip slides already processed (have tiles in output dir)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: cpu_count). Use 1 for single-process debugging.'
    )

    args = parser.parse_args()

    # Find pairs
    pairs = find_tiff_geojson_pairs(args.input)
    if not pairs:
        logger.error(f"No TIFF+GeoJSON pairs found in {args.input}")
        return

    # Filter already-processed slides if --resume
    if args.resume:
        os.makedirs(args.output, exist_ok=True)
        already_done = [p for p in pairs if slide_already_processed(p['stem'], args.output)]
        pairs = [p for p in pairs if not slide_already_processed(p['stem'], args.output)]
        if already_done:
            logger.info(f"--resume: Skipping {len(already_done)} already-processed slides")
            for p in already_done:
                logger.info(f"  ✓ {p['stem']}")
        logger.info(f"Will process {len(pairs)} remaining slides")

    if args.max_slides:
        pairs = pairs[:args.max_slides]
        logger.info(f"Processing first {len(pairs)} slides (--max-slides)")

    # Get canonical class map (fixed, no scanning needed)
    class_map = get_canonical_class_map()

    # Schedule processing
    os.makedirs(args.output, exist_ok=True)
    results = dynamic_schedule(
        pairs,
        args.output,
        class_map,
        ram_fraction=args.ram_fraction,
        tile_size=args.tile_size,
        max_workers=args.workers
    )

    # Summary
    successes = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']

    logger.info(f"\n{'='*80}")
    logger.info(f"SUMMARY: {len(successes)} success, {len(errors)} errors")
    if errors:
        logger.error("Errors:")
        for r in errors:
            logger.error(f"  {r['slide_name']}: {r['message']}")

if __name__ == '__main__':
    main()
