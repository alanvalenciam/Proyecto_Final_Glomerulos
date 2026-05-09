"""Generate tiled training data for UNet from WSI TIFF + GeoJSON annotations."""

import os
import json
import yaml
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
    import tifffile
except ImportError:
    tifffile = None

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python is required: pip install opencv-python")


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

# Default RGB color map for masks (hex → RGB tuple)
DEFAULT_CLASS_COLORS = {
    'background':       '#000000',
    'No_Proliferativo': '#ee8718',
    'Proliferativo':    '#00ffff',
    'Esclerosado':      '#ff00ff',
    'Excluido':         '#00ff00',
}

LARGE_FILE_THRESHOLD = 300 * 1024 * 1024  # 300 MB

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
    'excluyente': 'Excluido',
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
    if hasattr(img_or_slide, 'get_thumbnail'):  # OpenSlide — memory-efficient
        w, h = img_or_slide.level_dimensions[0]
        thumb = img_or_slide.get_thumbnail((w // downsample_factor, h // downsample_factor))
    else:  # PIL Image or numpy.ndarray
        if isinstance(img_or_slide, np.ndarray):
            logger.warning("Otsu mask is not supported for numpy array inputs")
            return None, None
        w, h = img_or_slide.size
        try:
            if w > 20000 or h > 20000:
                crop_w = min(8000, w)
                crop_h = min(8000, h)
                thumb = img_or_slide.crop((0, 0, crop_w, crop_h))
                thumb.thumbnail((max(64, crop_w // 100), max(64, crop_h // 100)), Image.Resampling.LANCZOS)
            elif w > 10000 or h > 10000:
                crop_w = min(5000, w)
                crop_h = min(5000, h)
                thumb = img_or_slide.crop((0, 0, crop_w, crop_h))
                thumb.thumbnail((max(64, crop_w // 50), max(64, crop_h // 50)), Image.Resampling.LANCZOS)
            else:
                target_w = max(64, w // downsample_factor)
                target_h = max(64, h // downsample_factor)
                thumb = img_or_slide.copy()
                thumb.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        except MemoryError:
            logger.warning(f"MemoryError during Otsu crop/thumbnail, returning None")
            return None, None

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
    """Detect image format: 'openslide', 'pil', 'tifffile', or None."""
    ext = Path(file_path).suffix.lower()
    size = os.path.getsize(file_path)

    # For TIFF files, strongly prefer OpenSlide if available (WSI-efficient)
    if ext in {'.tif', '.tiff'}:
        if openslide and size > LARGE_FILE_THRESHOLD:
            return 'openslide'
        if size < LARGE_FILE_THRESHOLD:
            return 'pil'
        # Large TIFF without OpenSlide: use tifffile for memory-mapped access
        if not openslide and size > LARGE_FILE_THRESHOLD:
            if tifffile:
                logger.info(f"Large TIFF file ({size / 1e9:.1f} GB) without OpenSlide. Using tifffile for memory-mapped access.")
                return 'tifffile'
            else:
                logger.warning(f"Large TIFF file ({size / 1e9:.1f} GB) without OpenSlide or tifffile. Install 'tifffile': pip install tifffile")
                return 'pil'
        return 'pil'
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

    # tifffile branch (memory-mapped for large TIFFs)
    if fmt == 'tifffile':
        try:
            slide = tifffile.memmap(tiff_path)
            h, w = slide.shape[:2]  # memmap returns (height, width, ...)
            logger.info(f"Loaded {tiff_path} with tifffile.memmap: {w}x{h}")
            return slide, w, h, 'tifffile'
        except Exception as e:
            logger.error(f"tifffile.memmap failed for {tiff_path}: {e}. Falling back to PIL.")
            # Fallback to PIL as last resort
            try:
                img = Image.open(tiff_path)
                w, h = img.size
                return img, w, h, 'pil'
            except Exception as e2:
                logger.error(f"PIL also failed on {tiff_path}: {e2}")
                raise

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

def extract_tile_tifffile(slide: np.ndarray, x: int, y: int, size: int) -> Image.Image:
    """Extract a tile from a tifffile memmap."""
    h, w = slide.shape[:2]

    # Clamp to bounds
    x_start = max(0, min(x, w - 1))
    y_start = max(0, min(y, h - 1))
    x_end = min(x + size, w)
    y_end = min(y + size, h)

    # Extract region (memmap slice returns numpy array)
    region = slide[y_start:y_end, x_start:x_end]

    # Convert to PIL Image
    if len(region.shape) == 2:  # Grayscale
        tile = Image.fromarray(region, mode='L')
        # Convert to RGB for consistency
        tile = tile.convert('RGB')
    elif len(region.shape) == 3 and region.shape[2] == 3:  # RGB
        tile = Image.fromarray(region, mode='RGB')
    elif len(region.shape) == 3 and region.shape[2] == 4:  # RGBA
        tile = Image.fromarray(region, mode='RGBA')
        tile = tile.convert('RGB')
    else:
        # Fallback: treat as grayscale
        logger.warning(f"Unexpected shape in tifffile region: {region.shape}, treating as grayscale")
        if region.size > 0:
            tile = Image.fromarray(np.uint8(region), mode='L')
            tile = tile.convert('RGB')
        else:
            # Return white tile if region is empty
            tile = Image.new('RGB', (size, size), (255, 255, 255))
            return tile

    # Pad if needed (tile smaller than requested size)
    if tile.size != (size, size):
        padded = Image.new('RGB', (size, size), (255, 255, 255))
        padded.paste(tile, (0, 0))
        tile = padded

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

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string (#RRGGBB) to RGB tuple with validation."""
    # Strip whitespace and remove leading # if present
    hex_color = hex_color.strip()
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]

    # Validate: exactly 6 characters
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}. Expected 6 hex digits (e.g., 'ee8718').")

    # Validate: only hex digits (0-9, a-f, A-F)
    if not all(c in '0123456789abcdefABCDEF' for c in hex_color):
        raise ValueError(f"Invalid hex color: {hex_color}. Must contain only hex digits (0-9, a-f).")

    # Convert to RGB tuple
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_rgb_mask(
    binary_mask: np.ndarray,
    class_mask: np.ndarray,
    id_to_rgb: Dict[int, Tuple[int, int, int]],
    bg_rgb: Tuple[int, int, int]
) -> Image.Image:
    """Generate RGB mask from binary and class masks using precomputed color palette."""
    rgb_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[binary_mask == 0] = bg_rgb

    for class_id, rgb in id_to_rgb.items():
        mask_pixels = class_mask == class_id
        rgb_mask[mask_pixels] = rgb

    return Image.fromarray(rgb_mask, 'RGB')

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

def iter_polygons(geom):
    """Iterate over Polygon objects from any Shapely geometry (handles MultiPolygon, GeometryCollection, etc.)."""
    if geom.is_empty:
        return
    from shapely.geometry import MultiPolygon, GeometryCollection
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for g in geom.geoms:
            if isinstance(g, Polygon):
                yield g
            elif isinstance(g, MultiPolygon):
                for p in g.geoms:
                    yield p

def rasterize_polygon(
    polygon_coords: List[List[float]],
    tile_box: Tuple[int, int, int, int],
    tile_size: int = TILE_SIZE,
    zoom_scale: float = 1.0
) -> np.ndarray:
    """Rasterize a polygon to a numpy array within a tile's bounding box using cv2.fillPoly."""
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    x0, y0 = tile_box[0], tile_box[1]

    # Clip polygon to tile bounds with Shapely
    tile_poly = box(tile_box[0], tile_box[1], tile_box[2], tile_box[3])
    glom_poly = Polygon(polygon_coords)
    if not glom_poly.is_valid:
        glom_poly = glom_poly.buffer(0)
    if glom_poly.is_empty or glom_poly.area == 0:
        return mask

    clipped = glom_poly.intersection(tile_poly)
    if clipped.is_empty:
        return mask

    # Iterate over all polygons in the clipped result (handles MultiPolygon, GeometryCollection, etc.)
    for poly in iter_polygons(clipped):
        coords = list(poly.exterior.coords)

        # Convert native coords to output pixel coords
        pts = np.array([
            [int((c[0] - x0) * zoom_scale), int((c[1] - y0) * zoom_scale)]
            for c in coords
        ], dtype=np.int32)

        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], color=255)

    return mask


def compute_coverage(
    polygon_coords: List[List[float]],
    tile_box: Tuple[int, int, int, int]
) -> float:
    """Compute percentage of polygon area that overlaps with tile_box."""
    try:
        polygon = polygon_to_shapely(polygon_coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area == 0:
            return 0.0

        tile_poly = box(tile_box[0], tile_box[1], tile_box[2], tile_box[3])
        intersection = polygon.intersection(tile_poly)

        return (intersection.area / polygon.area) * 100.0
    except Exception as e:
        logger.warning(f"Failed to compute coverage: {e}")
        return 0.0

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
    class_colors: Dict[str, str],
    tile_size: int = TILE_SIZE,
    zoom_scale: float = ZOOM_SCALE,
    stride: int = STRIDE,
    min_coverage_pct: float = MIN_COVERAGE_PCT,
    output_format: str = OUTPUT_FORMAT,
    compression: int = COMPRESSION,
    otsu_min_ratio: float = OTSU_MIN_RATIO,
    bg_threshold: float = BG_THRESHOLD,
    downsample_factor: int = DOWNSAMPLE_FACTOR
) -> Dict:
    """Process a single TIFF+GeoJSON pair and generate tiles, masks, and metadata."""
    stem = pair['stem']
    image_path = pair['image_path']
    geojson_path = pair['geojson_path']

    try:
        logger.info(f"[{stem}] Loading TIFF and GeoJSON...")
        slide, width, height, fmt = load_slide(image_path)
        geojson_data = load_geojson(geojson_path)

        # Generate Otsu tissue mask for filtering (OpenSlide only; tifffile uses per-tile background filter)
        if fmt == "openslide":
            logger.info(f"[{stem}] Computing Otsu tissue mask...")
            otsu_mask, ds_factor = generate_otsu_mask(slide, downsample_factor=downsample_factor)
        else:
            logger.info(f"[{stem}] Skipping Otsu mask for fmt={fmt}; using per-tile background filter only")
            otsu_mask, ds_factor = None, None
        native_tile_size = int(tile_size / zoom_scale)
        native_stride = int(stride / zoom_scale)

        # Extract glomeruli from GeoJSON with class normalization
        glomeruli = []
        for feat_idx, feature in enumerate(geojson_data.get('features', [])):
            geom = feature.get('geometry', {})
            geom_type = geom.get('type')
            if geom_type == 'Polygon':
                polygon_coord_list = [geom['coordinates'][0]]
            elif geom_type == 'MultiPolygon':
                polygon_coord_list = [poly[0] for poly in geom['coordinates']]
            else:
                logger.debug(f"[{stem}] Feature {feat_idx} has unsupported geometry type {geom_type!r}, skipping")
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

            # Add each polygon as a separate glomerulus entry (with the same class)
            for coords in polygon_coord_list:
                if len(coords) < 3:
                    logger.warning(f"[{stem}] Feature {feat_idx} polygon has < 3 coordinates, skipping")
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
        grid_tiles = generate_grid_tiles(width, height, native_tile_size, native_stride)
        logger.info(f"[{stem}] Generated {len(grid_tiles)} grid tiles")

        # Compute primary/secondary assignments
        logger.info(f"[{stem}] Computing primary/secondary tile assignments...")
        assignments = compute_primary_secondary(glomeruli, grid_tiles, coverage_threshold=min_coverage_pct)

        # Determine which centered tiles are needed
        centered_glom_ids = [gid for gid, asg in assignments.items() if asg['needs_centered_tile']]
        logger.info(f"[{stem}] {len(centered_glom_ids)} glomeruli need centered tiles")

        # Create output directories
        slide_output_dir = Path(output_dir) / stem
        images_dir = slide_output_dir / 'images'
        masks_dir = slide_output_dir / 'masks'
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Precompute color palette once per slide (not per tile)
        id_to_name = {v: k for k, v in class_map.items()}
        id_to_rgb = {}
        for class_id, class_name in id_to_name.items():
            hex_color = class_colors.get(class_name, '#000000')
            id_to_rgb[class_id] = hex_to_rgb(hex_color)
        bg_rgb = hex_to_rgb(class_colors.get('background', '#000000'))

        # Build tile registry for metadata
        tile_registry = []

        # Process grid tiles with filtering
        logger.info(f"[{stem}] Extracting {len(grid_tiles)} grid tiles with tissue filtering...")
        tiles_saved = 0
        tiles_skipped = 0

        for tile_idx, (x, y, x_end, y_end) in enumerate(grid_tiles):
            # New tile naming: native coordinates with 5-digit padding
            tile_name = f"{stem}_tile_x{x:05d}_y{y:05d}_endx{x_end:05d}_endy{y_end:05d}"
            if fmt == 'openslide':
                tile_img = extract_tile_openslide(slide, x, y, native_tile_size)
            elif fmt == 'tifffile':
                tile_img = extract_tile_tifffile(slide, x, y, native_tile_size)
            else:
                tile_img = extract_tile_pil(slide, x, y, native_tile_size)

            # Resize to output tile_size if needed
            if tile_img.size != (tile_size, tile_size):
                tile_img = tile_img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

            binary_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            class_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            tile_glomeruli = []

            for glom_data in glomeruli:
                cov = compute_coverage(glom_data['coordinates'], (x, y, x_end, y_end))
                if cov > 0:
                    glom_mask = rasterize_polygon(
                        glom_data['coordinates'],
                        (x, y, x_end, y_end),
                        tile_size,
                        zoom_scale
                    )
                    # Vectorized binary union
                    np.maximum(binary_mask, glom_mask, out=binary_mask)
                    # Set class_id where annotation exists
                    class_id = glom_data['class_id']
                    class_mask[glom_mask > 0] = class_id

                    glom_id = glom_data['id']

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
                if not has_tissue_in_mask(otsu_mask, x, y, native_tile_size, ds_factor, otsu_min_ratio):
                    tiles_skipped += 1
                    continue
                if is_mostly_background(tile_img, bg_threshold):
                    tiles_skipped += 1
                    continue

            # Save tile image
            img_path = images_dir / f"{tile_name}.{output_format}"
            (Image.fromarray(np.array(tile_img)) if isinstance(tile_img, np.ndarray) else tile_img).save(str(img_path), output_format.upper(), compress_level=compression)

            # Generate and save RGB-colored mask
            rgb_mask = generate_rgb_mask(binary_mask, class_mask, id_to_rgb, bg_rgb)
            mask_path = masks_dir / f"{tile_name}_mask.png"
            rgb_mask.save(mask_path, 'PNG', compress_level=compression)

            tiles_saved += 1

            # Record in registry
            tile_registry.append({
                'type': 'grid',
                'image': str(img_path.relative_to(slide_output_dir)),
                'mask': str(mask_path.relative_to(slide_output_dir)),
                'origin_native': [x, y],
                'bbox_native': [x, y, x_end, y_end],
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
            x = int(max(0, cx - native_tile_size / 2))
            y = int(max(0, cy - native_tile_size / 2))
            x = max(0, min(x, width - native_tile_size))
            y = max(0, min(y, height - native_tile_size))
            x_end, y_end = x + native_tile_size, y + native_tile_size

            # Calculate real coverage in the centered tile (not placeholder 0.0)
            update_coverage_for_centered_tile(glom_id, glom_data['coordinates'], (x, y, x_end, y_end), assignments)

            tile_name = f"{stem}_centered_g{glom_id:04d}"
            if fmt == 'openslide':
                tile_img = extract_tile_openslide(slide, x, y, native_tile_size)
            elif fmt == 'tifffile':
                tile_img = extract_tile_tifffile(slide, x, y, native_tile_size)
            else:
                tile_img = extract_tile_pil(slide, x, y, native_tile_size)

            # Resize to output tile_size if needed
            if tile_img.size != (tile_size, tile_size):
                tile_img = tile_img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

            binary_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            class_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            tile_glomeruli = []

            for gdata in glomeruli:
                cov = compute_coverage(gdata['coordinates'], (x, y, x_end, y_end))
                if cov > 0:
                    glom_mask = rasterize_polygon(
                        gdata['coordinates'],
                        (x, y, x_end, y_end),
                        tile_size,
                        zoom_scale
                    )
                    # Vectorized binary union
                    np.maximum(binary_mask, glom_mask, out=binary_mask)
                    # Set class_id where annotation exists
                    class_id = gdata['class_id']
                    class_mask[glom_mask > 0] = class_id

                    gid = gdata['id']

                    role = 'primary' if gid == glom_id else 'secondary'
                    tile_glomeruli.append({
                        'id': gid,
                        'class': gdata['class_name'],
                        'coverage_pct': cov,
                        'role': role
                    })

            # Save tile image
            img_path = images_dir / f"{tile_name}.{output_format}"
            (Image.fromarray(np.array(tile_img)) if isinstance(tile_img, np.ndarray) else tile_img).save(str(img_path), output_format.upper(), compress_level=compression)

            # Generate and save RGB-colored mask
            rgb_mask = generate_rgb_mask(binary_mask, class_mask, id_to_rgb, bg_rgb)
            mask_path = masks_dir / f"{tile_name}_mask.png"
            rgb_mask.save(mask_path, 'PNG', compress_level=compression)

            tile_registry.append({
                'type': 'centered',
                'image': str(img_path.relative_to(slide_output_dir)),
                'mask': str(mask_path.relative_to(slide_output_dir)),
                'origin_native': [x, y],
                'bbox_native': [x, y, x_end, y_end],
                'glomerulus_id': glom_id,
                'has_annotations': len(tile_glomeruli) > 0,
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
            'class_map': class_map,
            'class_colors': class_colors,
            'n_glomeruli': len(glomeruli),
            'n_grid_tiles': len(grid_tiles),
            'n_centered_tiles': len(centered_glom_ids),
            'otsu_applied': otsu_mask is not None,
            'glomeruli_coverage': glomeruli_coverage,
            'tiling_config': {
                'tile_size': tile_size,
                'zoom_scale': zoom_scale,
                'stride': stride,
                'min_coverage_pct': min_coverage_pct,
                'output_format': output_format,
                'compression': compression,
                'otsu_min_ratio': otsu_min_ratio,
                'bg_threshold': bg_threshold,
                'downsample_factor': downsample_factor,
            },
            'tiles': tile_registry
        }

        annotations_path = slide_output_dir / 'annotations.json'
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)

        # Write _SUCCESS sentinel to mark complete processing
        (slide_output_dir / "_SUCCESS").write_text("ok\n")

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
    class_colors: Dict[str, str],
    ram_fraction: float = 0.5,
    tile_size: int = TILE_SIZE,
    zoom_scale: float = ZOOM_SCALE,
    stride: int = STRIDE,
    max_workers: int = None,
    min_coverage_pct: float = MIN_COVERAGE_PCT,
    output_format: str = OUTPUT_FORMAT,
    compression: int = COMPRESSION,
    otsu_min_ratio: float = OTSU_MIN_RATIO,
    bg_threshold: float = BG_THRESHOLD,
    downsample_factor: int = DOWNSAMPLE_FACTOR
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
                    class_colors,
                    tile_size,
                    zoom_scale,
                    stride,
                    min_coverage_pct,
                    output_format,
                    compression,
                    otsu_min_ratio,
                    bg_threshold,
                    downsample_factor
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

def validate_class_colors(class_map: Dict[str, int], class_colors: Dict[str, str]) -> None:
    """Validate that all classes in class_map have corresponding colors defined."""
    missing = [name for name in class_map.keys() if name not in class_colors]
    if missing:
        raise ValueError(f"Missing colors for classes: {missing}")

def slide_already_processed(stem: str, output_dir: str) -> bool:
    """Check if a slide has been fully processed (marked by _SUCCESS sentinel)."""
    return (Path(output_dir) / stem / "_SUCCESS").exists()

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file. Falls back to defaults if file not found."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config if config else {}
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except Exception as e:
        logger.warning(f"Error loading config file {config_path}: {e}, using defaults")
        return {}

def main():
    parser = argparse.ArgumentParser(
        description='Generate tiled training data for UNet from WSI + GeoJSON annotations'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config file (YAML)'
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
        default=None,
        help='Tile size in pixels (at zoom_scale)'
    )
    parser.add_argument(
        '--zoom-scale',
        type=float,
        default=None,
        help='Zoom scale (1 tile pixel = 1/zoom_scale native pixels)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=None,
        help='Stride between tiles (default = tile_size - overlap for 50% overlap)'
    )
    parser.add_argument(
        '--ram-fraction',
        type=float,
        default=0.5,
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

    # Load configuration from YAML file
    config = load_config(args.config)

    # Merge config with CLI args (CLI args take precedence if explicitly provided)
    tile_size = args.tile_size if args.tile_size is not None else config.get('tile_size', TILE_SIZE)
    zoom_scale = args.zoom_scale if args.zoom_scale is not None else config.get('zoom_scale', ZOOM_SCALE)
    stride = args.stride if args.stride is not None else config.get('stride', STRIDE)
    min_coverage_pct = config.get('min_coverage_pct', MIN_COVERAGE_PCT)
    output_format = config.get('output_format', OUTPUT_FORMAT)
    compression = config.get('compression', COMPRESSION)

    # Load tissue filter parameters from config
    tissue_filter_cfg = config.get('tissue_filter', {})
    otsu_min_ratio = tissue_filter_cfg.get('otsu_min_ratio', OTSU_MIN_RATIO)
    bg_threshold = tissue_filter_cfg.get('bg_threshold', BG_THRESHOLD)
    downsample_factor = tissue_filter_cfg.get('downsample_factor', DOWNSAMPLE_FACTOR)

    # Load class map from config (if provided), otherwise use canonical
    class_map = config.get('class_map')
    if not class_map:
        class_map = get_canonical_class_map()
    else:
        logger.info(f"Using class map from config: {class_map}")

    # Load class colors from config for RGB masks
    class_colors = config.get('class_colors', DEFAULT_CLASS_COLORS)
    logger.info(f"Using class colors: {class_colors}")

    # Validate that all classes have colors
    validate_class_colors(class_map, class_colors)

    logger.info(f"Config: tile_size={tile_size}, zoom_scale={zoom_scale}, stride={stride}")
    logger.info(f"Tissue filter: otsu_min_ratio={otsu_min_ratio}, bg_threshold={bg_threshold}, downsample_factor={downsample_factor}")

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

    # Schedule processing
    os.makedirs(args.output, exist_ok=True)
    results = dynamic_schedule(
        pairs,
        args.output,
        class_map,
        class_colors,
        ram_fraction=args.ram_fraction,
        tile_size=tile_size,
        zoom_scale=zoom_scale,
        stride=stride,
        max_workers=args.workers,
        min_coverage_pct=min_coverage_pct,
        output_format=output_format,
        compression=compression,
        otsu_min_ratio=otsu_min_ratio,
        bg_threshold=bg_threshold,
        downsample_factor=downsample_factor
    )

    # Aggregate dataset summary from all annotations.json files
    dataset_summary = {
        'n_slides': 0,
        'n_tiles_total': 0,
        'n_tiles_positive': 0,
        'n_tiles_negative': 0,
        'class_counts': {}
    }

    output_path = Path(args.output)
    for slide_dir in output_path.iterdir():
        if not slide_dir.is_dir():
            continue
        if not (slide_dir / '_SUCCESS').exists():
            continue
        annotations_file = slide_dir / 'annotations.json'
        if annotations_file.exists():
            try:
                with open(annotations_file, 'r') as f:
                    annotations = json.load(f)

                dataset_summary['n_slides'] += 1

                if 'tiles' in annotations:
                    for tile in annotations['tiles']:
                        dataset_summary['n_tiles_total'] += 1
                        if tile.get('has_annotations', False):
                            dataset_summary['n_tiles_positive'] += 1
                        else:
                            dataset_summary['n_tiles_negative'] += 1

                if 'glomeruli_coverage' in annotations:
                    for _, glom_data in annotations['glomeruli_coverage'].items():
                        class_name = glom_data.get('class', 'unknown')
                        dataset_summary['class_counts'][class_name] = dataset_summary['class_counts'].get(class_name, 0) + 1
            except Exception as e:
                logger.warning(f"Failed to read {annotations_file}: {e}")

    summary_path = output_path / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    logger.info(f"Wrote dataset summary to {summary_path}")

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
