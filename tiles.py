import os
import json
import logging
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
from typing import List, Dict, Tuple, Optional

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    logger.warning("openslide-python not installed. Install with: pip install openslide-python. SVS, NDPI, and VMS formats will not be supported.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("opencv-python not installed. Install with: pip install opencv-python. Otsu-based tissue masking will be disabled.")

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    logger.warning("tifffile not installed. Large TIFFs will fall back to PIL. Install: pip install tifffile")

OPENSLIDE_FORMATS = {'.svs', '.ndpi', '.vms'}
PIL_FORMATS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
ALL_FORMATS = OPENSLIDE_FORMATS | PIL_FORMATS

LARGE_FILE_THRESHOLD = 300 * 1024 * 1024  # 300 MB — route large TIFFs away from PIL


def _safe_worker_count(image_paths):
    """
    Compute safe worker count based on available RAM and image sizes.

    TIFFs decompress to ~4x their file size in memory.
    Uses psutil if available, falls back to conservative estimate.
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        available_gb = 4.0  # Conservative fallback

    max_file_size_gb = 0.0
    for path in image_paths:
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            max_file_size_gb = max(max_file_size_gb, size_gb)

    if max_file_size_gb == 0:
        max_file_size_gb = 1.0

    expansion_factor = 4  # TIFFs decompress to ~4x
    memory_per_worker = max_file_size_gb * expansion_factor

    if memory_per_worker > 0:
        num_workers = int((available_gb * 0.80) / memory_per_worker)
        num_workers = max(1, min(num_workers, cpu_count()))
    else:
        num_workers = min(4, cpu_count())

    logger.info(f"RAM-aware: {available_gb:.2f}GB available, max file {max_file_size_gb:.3f}GB → {num_workers} workers")
    return num_workers


def detect_format(file_path):
    """Detect image format by file extension and size. Routes large TIFFs to OpenSlide or tifffile."""
    ext = os.path.splitext(file_path)[1].lower()
    size = os.path.getsize(file_path)

    # For TIFF files, route by size and backend availability
    if ext in {'.tif', '.tiff'}:
        if OPENSLIDE_AVAILABLE and size > LARGE_FILE_THRESHOLD:
            return 'openslide'
        if size < LARGE_FILE_THRESHOLD:
            return 'pil'
        if not OPENSLIDE_AVAILABLE and size > LARGE_FILE_THRESHOLD:
            if TIFFFILE_AVAILABLE:
                logger.info(f"Large TIFF ({size / 1e9:.1f} GB) without OpenSlide — using tifffile memmap")
                return 'tifffile'
            else:
                logger.warning("Large TIFF without OpenSlide or tifffile — attempting PIL (may OOM)")
                return 'pil'
        return 'pil'

    # OpenSlide formats
    if ext in OPENSLIDE_FORMATS and OPENSLIDE_AVAILABLE:
        return 'openslide'
    # Regular image formats
    if ext in {'.png', '.jpg', '.jpeg'}:
        return 'pil'
    return None


def generate_otsu_mask(img_or_slide, downsample_factor=20):
    """Generate binary tissue mask via Otsu thresholding on HSV saturation."""
    if not OPENCV_AVAILABLE:
        return None, None

    if hasattr(img_or_slide, 'get_thumbnail'):  # OpenSlide
        w, h = img_or_slide.level_dimensions[0]
        thumb = img_or_slide.get_thumbnail((w // downsample_factor, h // downsample_factor))
    elif isinstance(img_or_slide, np.ndarray):  # tifffile memmap
        logger.info("Otsu mask skipped for tifffile/numpy input — using per-tile background filter only")
        return None, None
    else:  # PIL Image
        w, h = img_or_slide.size
        try:
            thumb = img_or_slide.resize((w // downsample_factor, h // downsample_factor), Image.Resampling.LANCZOS)
        except MemoryError:
            logger.warning("MemoryError generating Otsu mask — skipping (per-tile filter still active)")
            return None, None

    thumb_cv = cv2.cvtColor(np.array(thumb.convert('RGB')), cv2.COLOR_RGB2HSV)
    saturation = cv2.medianBlur(thumb_cv[:, :, 1], 7)
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, downsample_factor


def has_tissue_in_mask(mask, x, y, tile_size, downsample_factor, min_ratio=0.01):
    """Fast pre-check: does tile region contain tissue via Otsu mask?

    Args:
        min_ratio: Minimum tissue ratio to keep tile (default 0.01 = 1%).
                  Use 0.01-0.02 for normal quality, 0.005-0.01 for low-quality staining.
    """
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


def is_mostly_background(tile, min_std=15.0):
    """Return True if tile lacks tissue (should be skipped)."""
    gray = np.array(tile.convert('L'))
    return float(np.std(gray)) < min_std


def load_geojson_annotations(slide_name: str, annotations_dir: str) -> Optional[List[Dict]]:
    """
    Load GeoJSON annotations for a slide.

    Args:
        slide_name: Biopsy/slide identifier (e.g., '18-139')
        annotations_dir: Path to directory containing GeoJSON files

    Returns:
        List of annotation dicts with keys: gid, class, color, polygon
        or None if no file found
    """
    annotations_dir = Path(annotations_dir)
    if not annotations_dir.exists():
        return None

    geojson_path = annotations_dir / f"{slide_name}.geojson"
    if not geojson_path.exists():
        candidates = list(annotations_dir.glob(f"*{slide_name}*.geojson"))
        if candidates:
            geojson_path = candidates[0]
        else:
            return None

    try:
        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load annotations from {geojson_path}: {e}")
        return None

    annotations = []
    features = geojson.get('features', [])

    for gid, feature in enumerate(features):
        geom = feature.get('geometry', {})
        props = feature.get('properties', {})

        if geom.get('type') != 'Polygon':
            continue

        coords = geom.get('coordinates', [[]])
        if not coords or not coords[0]:
            continue

        class_name = props.get('classification', {}).get('name', 'unknown')
        color = props.get('classification', {}).get('color', [0, 0, 0])

        annotations.append({
            'gid': gid,
            'class': class_name,
            'color': color,
            'polygon': coords[0],  # Exterior ring only
        })

    return annotations if annotations else None


def polygon_intersects_tile(polygon: List[Tuple], tile_x: int, tile_y: int, tile_size: int) -> bool:
    """Check if a polygon intersects with a tile's bounding box (all in native coords)."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    poly_minx, poly_miny = min(xs), min(ys)
    poly_maxx, poly_maxy = max(xs), max(ys)

    tile_minx, tile_miny = tile_x, tile_y
    tile_maxx, tile_maxy = tile_x + tile_size, tile_y + tile_size

    return not (poly_maxx < tile_minx or poly_minx > tile_maxx or
                poly_maxy < tile_miny or poly_miny > tile_maxy)


def class_name_to_gray_value(class_name: str) -> int:
    """Map glomerulus class name to grayscale value.

    Values from config.yaml:
    - background: 0
    - No_Proliferativo: 64
    - Proliferativo: 128
    - Esclerosado: 192
    - Excluido: 255
    """
    class_map = {
        'no prolif': 64,           # GeoJSON often uses lowercase/abbreviated
        'No_Proliferativo': 64,
        'prolif': 128,
        'Proliferativo': 128,
        'esclerosado': 192,
        'Esclerosado': 192,
        'exclude': 255,
        'Excluido': 255,
        'excluded': 255,
    }
    # Case-insensitive lookup
    for key, value in class_map.items():
        if key.lower() == class_name.lower():
            return value
    # Default to 64 (No_Proliferativo) if not found
    logger.debug(f"Unknown class '{class_name}', defaulting to 64 (No_Proliferativo)")
    return 64


def rasterize_polygon_to_mask(polygon: List[Tuple], tile_x: int, tile_y: int,
                              native_tile_size: int, output_tile_size: int,
                              zoom_scale: float, gray_value: int = 255) -> np.ndarray:
    """
    Rasterize a single polygon to a mask at native resolution, then resize.

    Args:
        polygon: List of (x, y) tuples in native slide coordinates
        tile_x, tile_y: Native tile top-left corner
        native_tile_size: Tile size in native coordinates
        output_tile_size: Output tile size (e.g., 1024)
        zoom_scale: Downsampling factor (e.g., 0.5)
        gray_value: Grayscale value to fill polygon (0-255)

    Returns:
        Numpy array (output_tile_size × output_tile_size, dtype uint8, 0 or gray_value)
    """
    mask = Image.new('L', (native_tile_size, native_tile_size), 0)
    draw = ImageDraw.Draw(mask)

    polygon_relative = [(p[0] - tile_x, p[1] - tile_y) for p in polygon]

    clipped_polygon = []
    for x, y in polygon_relative:
        x = max(0, min(x, native_tile_size))
        y = max(0, min(y, native_tile_size))
        clipped_polygon.append((x, y))

    if len(clipped_polygon) >= 3:
        try:
            draw.polygon(clipped_polygon, fill=gray_value)
        except Exception as e:
            logger.debug(f"Failed to rasterize polygon: {e}")

    mask_array = np.array(mask, dtype=np.uint8)

    if output_tile_size != native_tile_size:
        scale_factor = output_tile_size / native_tile_size
        mask_pil = Image.fromarray(mask_array, mode='L')
        new_size = (output_tile_size, output_tile_size)
        mask_pil = mask_pil.resize(new_size, Image.Resampling.NEAREST)
        mask_array = np.array(mask_pil, dtype=np.uint8)

    return mask_array


def rasterize_annotations_to_tile_mask(
    annotations: List[Dict], tile_x: int, tile_y: int,
    tile_size: int, native_tile_size: int, zoom_scale: float
) -> Tuple[Image.Image, List[int]]:
    """
    Rasterize all annotations intersecting a tile to a single mask image.
    Uses grayscale values from class_name_to_gray_value() to encode class information.

    Args:
        annotations: List of annotation dicts (from load_geojson_annotations)
        tile_x, tile_y: Native tile top-left corner
        tile_size: Output tile size (e.g., 1024)
        native_tile_size: Tile size in native coordinates
        zoom_scale: Downsampling factor

    Returns:
        (PIL Image 'L' grayscale with class values, list of GIDs present in tile)
    """
    if not annotations:
        return Image.new('L', (tile_size, tile_size), 0), []

    # Start with output size mask (already resized)
    combined_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    tile_gids = []

    for ann in annotations:
        if not polygon_intersects_tile(ann['polygon'], tile_x, tile_y, native_tile_size):
            continue

        # Get grayscale value for this glomerulus class
        gray_value = class_name_to_gray_value(ann['class'])

        poly_mask = rasterize_polygon_to_mask(
            ann['polygon'], tile_x, tile_y,
            native_tile_size, tile_size, zoom_scale,
            gray_value=gray_value  # Pass class-specific gray value
        )
        # poly_mask is already (tile_size, tile_size), so no reshape needed
        # Use maximum to preserve highest class value if polygons overlap
        combined_mask = np.maximum(combined_mask, poly_mask)
        tile_gids.append(ann['gid'])

    return Image.fromarray(combined_mask, mode='L'), tile_gids


def build_glomerulus_tile_index(
    annotations: List[Dict], tile_records: List[Dict]
) -> Dict:
    """
    Build index mapping glomeruli to their containing tiles.

    Args:
        annotations: List of annotation dicts
        tile_records: List of dicts with keys 'tile' (filename), 'gids' (list of GIDs)

    Returns:
        Dict with 'glomeruli' and 'tile_to_glomeruli' keys
    """
    tile_to_glomeruli = {}
    glomeruli_tiles = {}

    for record in tile_records:
        tile_name = record['tile']
        gids = record['gids']
        if gids:
            tile_to_glomeruli[tile_name] = gids
            for gid in gids:
                glomeruli_tiles.setdefault(gid, []).append(tile_name)

    glomeruli_list = []
    for ann in annotations:
        gid = ann['gid']
        tiles = glomeruli_tiles.get(gid, [])
        glomeruli_list.append({
            'gid': gid,
            'class': ann['class'],
            'color': ann['color'],
            'tiles': tiles,
        })

    return {
        'glomeruli': glomeruli_list,
        'tile_to_glomeruli': tile_to_glomeruli,
    }


def save_tile(tile, tile_path, output_format='png'):
    """Save tile in specified format."""
    output_format = output_format.lower().strip('.')

    try:
        if output_format == 'png':
            tile.save(tile_path, 'PNG')
        elif output_format in ('jpg', 'jpeg'):
            if tile.mode != 'RGB':
                tile = tile.convert('RGB')
            tile.save(tile_path, 'JPEG', quality=95)
        elif output_format == 'tiff':
            tile.save(tile_path, 'TIFF', compression='lzw')
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    except Exception:
        if output_format == 'png':
            logger.warning("PNG save failed, falling back to JPEG")
            tile_path = tile_path.replace('.png', '.jpg')
            if tile.mode != 'RGB':
                tile = tile.convert('RGB')
            tile.save(tile_path, 'JPEG', quality=95)
        else:
            raise


def extract_tile_tifffile(slide, x, y, size):
    """Extract tile from tifffile memmap — reads only the requested region."""
    h, w = slide.shape[:2]
    x_start = max(0, min(x, w - 1))
    y_start = max(0, min(y, h - 1))
    x_end = min(x + size, w)
    y_end = min(y + size, h)
    region = slide[y_start:y_end, x_start:x_end]

    if len(region.shape) == 2:
        tile = Image.fromarray(region, mode='L').convert('RGB')
    elif len(region.shape) == 3 and region.shape[2] == 3:
        tile = Image.fromarray(region, mode='RGB')
    elif len(region.shape) == 3 and region.shape[2] == 4:
        tile = Image.fromarray(region, mode='RGBA').convert('RGB')
    else:
        tile = Image.new('RGB', (size, size), (255, 255, 255))
        return tile

    if tile.size != (size, size):
        padded = Image.new('RGB', (size, size), (255, 255, 255))
        padded.paste(tile, (0, 0))
        tile = padded
    return tile


def _process_single_image(args):
    """
    Process a single image to tiles. Worker function for multiprocessing.

    Args:
        args: Tuple of (file_name, input_dir, output_dir, tile_size, overlap,
                        target_mpp, zoom_scale, bg_threshold, output_format,
                        openslide_level, otsu_min_ratio, annotations_dir)

    Returns:
        str: base_name if successful, None if failed.
    """
    (file_name, input_dir, output_dir, tile_size, overlap, target_mpp,
     zoom_scale, bg_threshold, output_format, openslide_level, otsu_min_ratio,
     annotations_dir) = args

    image_path = os.path.join(input_dir, file_name)
    base_name = os.path.splitext(file_name)[0]
    file_ext = os.path.splitext(file_name)[1].lower()

    slide_output_dir = os.path.join(output_dir, "Imagen", base_name)
    image_output_dir = os.path.join(slide_output_dir, "images")
    mask_output_dir = os.path.join(slide_output_dir, "masks") if annotations_dir else None

    Path(image_output_dir).mkdir(parents=True, exist_ok=True)
    if mask_output_dir:
        Path(mask_output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing: {file_name}")
    logger.info(f"  Type: {file_ext[1:].upper()}")
    logger.info(f"  Output dir: {image_output_dir}")

    annotations = None
    tile_records = []
    if annotations_dir:
        annotations = load_geojson_annotations(base_name, annotations_dir)
        if annotations:
            logger.info(f"  Loaded {len(annotations)} annotations from GeoJSON")

    try:
        handler = detect_format(image_path)
        mask = None
        downsample_factor = None
        metadata = {}

        if handler == 'openslide':
            slide = openslide.open_slide(image_path)
            native_width, native_height = slide.level_dimensions[0]
            level_count = slide.level_count
            logger.info(f"  Dimensions (level 0): {native_width}x{native_height}")
            logger.info(f"  Resolution levels: {level_count}")
            logger.info(f"  Tiling with zoom_scale={zoom_scale} (native capture × {1/zoom_scale:.1f})")

            # Generate Otsu mask
            if OPENCV_AVAILABLE:
                mask, downsample_factor = generate_otsu_mask(slide, downsample_factor=20)
                logger.info(f"  Otsu mask generated (downsample × {downsample_factor})")

            # Compute iteration parameters in native space using zoom_scale
            native_tile_size = int(tile_size / zoom_scale)
            native_stride = int((tile_size - overlap) / zoom_scale)

            saved_count = 0
            skipped_count = 0

            # Iterate in NATIVE (level 0) coordinate space
            for y in range(0, native_height, native_stride):
                for x in range(0, native_width, native_stride):
                    native_x = x
                    native_y = y
                    native_x_end = min(x + native_tile_size, native_width)
                    native_y_end = min(y + native_tile_size, native_height)

                    # Fast pre-check via Otsu mask
                    if mask is not None and not has_tissue_in_mask(mask, native_x, native_y, native_tile_size, downsample_factor, min_ratio=otsu_min_ratio):
                        skipped_count += 1
                        continue

                    # Read tile from slide at native resolution
                    tile_pil = slide.read_region((native_x, native_y), 0, (native_tile_size, native_tile_size))
                    if tile_pil.mode == 'RGBA':
                        tile = Image.new('RGB', tile_pil.size, (255, 255, 255))
                        tile.paste(tile_pil, mask=tile_pil.split()[3])
                    else:
                        tile = tile_pil.convert('RGB') if tile_pil.mode != 'RGB' else tile_pil

                    # Resize to output tile_size (at target_mpp)
                    if tile.size != (tile_size, tile_size):
                        tile = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

                    # Secondary filter: grayscale std
                    if is_mostly_background(tile, bg_threshold):
                        skipped_count += 1
                        continue

                    # Save tile with NATIVE coordinates in filename
                    ext = '.' + output_format.lstrip('.')
                    tile_name = f"{base_name}_tile_x{native_x:05d}_y{native_y:05d}_endx{native_x_end:05d}_endy{native_y_end:05d}{ext}"
                    tile_path = os.path.join(image_output_dir, tile_name)

                    # Pad tile to fixed size with white background
                    if tile.size != (tile_size, tile_size):
                        padded_tile = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
                        padded_tile.paste(tile, (0, 0))
                        tile = padded_tile

                    save_tile(tile, tile_path, output_format)

                    # Save mask if annotations available
                    if annotations:
                        mask_img, tile_gids = rasterize_annotations_to_tile_mask(
                            annotations, native_x, native_y,
                            tile_size, native_tile_size, zoom_scale
                        )
                        mask_path = os.path.join(mask_output_dir, tile_name.replace(ext, '_mask.png'))
                        mask_img.save(mask_path, 'PNG')
                        tile_records.append({'tile': tile_name, 'gids': tile_gids})

                    saved_count += 1

                    if saved_count % 100 == 0:
                        logger.info(f"  Saved {saved_count} tiles...")

            logger.info(f"  Final: {saved_count} tiles saved, {skipped_count} background tiles skipped")

            # Save metadata
            metadata = {
                "tile_size": tile_size,
                "overlap": overlap,
                "zoom_scale": zoom_scale,
                "native_width": native_width,
                "native_height": native_height,
                "format": "openslide",
                "has_annotations": annotations is not None,
                "glomerulus_count": len(annotations) if annotations else 0
            }

        elif handler == 'pil':
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                rgb_image = Image.new('RGB', img.size, (255, 255, 255))
                rgb_image.paste(img, mask=img.split()[3])
                img = rgb_image
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            native_width, native_height = img.size
            logger.info(f"  Dimensions: {native_width}x{native_height}")

            # PIL images: tile using zoom_scale (no MPP metadata available)
            # DPI metadata in TIFF is often screen resolution (96 DPI), not microscopy calibration
            # Users should provide calibrated WSI files (SVS/NDPI) for MPP-aware tiling
            native_mpp = None
            downsample_ratio = zoom_scale
            logger.info(f"  Tiling with zoom_scale={zoom_scale} (native capture × {1/zoom_scale:.1f})")
            logger.info("Tip: For MPP-aware tiling, use OpenSlide formats (SVS, NDPI, VMS)")

            # Generate Otsu mask
            if OPENCV_AVAILABLE:
                mask, downsample_factor = generate_otsu_mask(img, downsample_factor=20)
                logger.info(f"  Otsu mask generated (downsample × {downsample_factor})")

            native_tile_size = int(tile_size / zoom_scale)
            native_stride = int((tile_size - overlap) / zoom_scale)
            saved_count = 0
            skipped_count = 0

            # Iterate in native (source) space
            for y in range(0, native_height, native_stride):
                for x in range(0, native_width, native_stride):
                    x_end = min(x + native_tile_size, native_width)
                    y_end = min(y + native_tile_size, native_height)

                    # Read larger region from original
                    tile = img.crop((x, y, x_end, y_end))

                    # Fast pre-check via Otsu mask
                    if mask is not None and not has_tissue_in_mask(mask, x, y, native_tile_size, downsample_factor, min_ratio=otsu_min_ratio):
                        skipped_count += 1
                        continue

                    # Resize to output tile_size
                    if tile.size != (tile_size, tile_size):
                        tile = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

                    # Secondary filter: grayscale std (on resized tile)
                    if is_mostly_background(tile, bg_threshold):
                        skipped_count += 1
                        continue

                    # Save tile with NATIVE coordinates in filename
                    ext = '.' + output_format.lstrip('.')
                    tile_name = f"{base_name}_tile_x{x:05d}_y{y:05d}_endx{x_end:05d}_endy{y_end:05d}{ext}"
                    tile_path = os.path.join(image_output_dir, tile_name)

                    # Pad tile to fixed size if needed (edge tiles)
                    if tile.size != (tile_size, tile_size):
                        padded_tile = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
                        padded_tile.paste(tile, (0, 0))
                        tile = padded_tile

                    save_tile(tile, tile_path, output_format)

                    # Save mask if annotations available
                    if annotations:
                        mask_img, tile_gids = rasterize_annotations_to_tile_mask(
                            annotations, x, y,
                            tile_size, native_tile_size, zoom_scale
                        )
                        mask_path = os.path.join(mask_output_dir, tile_name.replace(ext, '_mask.png'))
                        mask_img.save(mask_path, 'PNG')
                        tile_records.append({'tile': tile_name, 'gids': tile_gids})

                    saved_count += 1

                    if saved_count % 100 == 0:
                        logger.info(f"  Saved {saved_count} tiles...")

            logger.info(f"  Final: {saved_count} tiles saved, {skipped_count} background tiles skipped")

            # Save metadata
            metadata = {
                "native_mpp": native_mpp,
                "target_mpp": None,
                "downsample_ratio": zoom_scale,
                "tile_size": tile_size,
                "overlap": overlap,
                "native_width": native_width,
                "native_height": native_height,
                "format": "pil",
                "zoom_scale": zoom_scale,
                "has_annotations": annotations is not None,
                "glomerulus_count": len(annotations) if annotations else 0
            }

        elif handler == 'tifffile':
            slide = tifffile.memmap(image_path)
            h, w = slide.shape[:2]
            native_width, native_height = w, h
            logger.info(f"  Dimensions: {native_width}x{native_height} (tifffile memmap)")

            native_mpp = None
            downsample_ratio = zoom_scale
            logger.info(f"  Tiling with zoom_scale={zoom_scale} (native capture × {1/zoom_scale:.1f})")

            # Otsu mask not supported for memmap — per-tile bg filter still active
            mask, downsample_factor = None, None

            native_tile_size = int(tile_size / zoom_scale)
            native_stride = int((tile_size - overlap) / zoom_scale)
            saved_count = 0
            skipped_count = 0

            for y in range(0, native_height, native_stride):
                for x in range(0, native_width, native_stride):
                    x_end = min(x + native_tile_size, native_width)
                    y_end = min(y + native_tile_size, native_height)

                    tile = extract_tile_tifffile(slide, x, y, native_tile_size)

                    if tile.size != (tile_size, tile_size):
                        tile = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)

                    if is_mostly_background(tile, bg_threshold):
                        skipped_count += 1
                        continue

                    ext = '.' + output_format.lstrip('.')
                    tile_name = f"{base_name}_tile_x{x:05d}_y{y:05d}_endx{x_end:05d}_endy{y_end:05d}{ext}"
                    tile_path = os.path.join(image_output_dir, tile_name)

                    save_tile(tile, tile_path, output_format)

                    # Save mask if annotations available
                    if annotations:
                        mask_img, tile_gids = rasterize_annotations_to_tile_mask(
                            annotations, x, y,
                            tile_size, native_tile_size, zoom_scale
                        )
                        mask_path = os.path.join(mask_output_dir, tile_name.replace(ext, '_mask.png'))
                        mask_img.save(mask_path, 'PNG')
                        tile_records.append({'tile': tile_name, 'gids': tile_gids})

                    saved_count += 1

                    if saved_count % 100 == 0:
                        logger.info(f"  Saved {saved_count} tiles...")

            logger.info(f"  Final: {saved_count} tiles saved, {skipped_count} background tiles skipped")

            metadata = {
                "native_mpp": None,
                "target_mpp": None,
                "downsample_ratio": zoom_scale,
                "tile_size": tile_size,
                "overlap": overlap,
                "native_width": native_width,
                "native_height": native_height,
                "format": "tifffile",
                "zoom_scale": zoom_scale,
                "has_annotations": annotations is not None,
                "glomerulus_count": len(annotations) if annotations else 0
            }

        else:
            raise ValueError(f"Unsupported format: {file_ext}")

        # Save metadata.json
        metadata_path = os.path.join(slide_output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  Metadata saved to metadata.json")

        # Save glomerulus index if annotations were processed
        if annotations and tile_records:
            index = build_glomerulus_tile_index(annotations, tile_records)
            index_path = os.path.join(slide_output_dir, "glomerulus_index.json")
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)
            logger.info(f"  Glomerulus index saved ({len(annotations)} glomeruli, {len(set(r['tile'] for r in tile_records))} tiles with annotations)")

        return base_name

    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}", exc_info=True)
        return None


def process_folder_to_subfolders(input_dir, output_dir, tile_size=1024, overlap=512,
                                target_mpp=0.5, zoom_scale=0.5, bg_threshold=15.0, output_format='png',
                                openslide_level=0, format_filter=None, otsu_min_ratio=0.01, num_workers=None,
                                annotations_dir=None):
    """
    Process histopathology images into tiles using MPP-aware resolution with parallel processing.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output tiles
        tile_size: Size of output tile in pixels (at target_mpp resolution)
        overlap: Overlap between tiles in pixels (at target_mpp resolution)
        target_mpp: Target resolution in µm/px for OpenSlide images (e.g., 0.5 means 20x magnification)
        zoom_scale: Downsampling scale for PIL images without MPP metadata (default 0.5 = 2x downsampling)
        bg_threshold: Minimum grayscale std deviation to keep tile (default 15.0)
        output_format: Output format: 'png', 'jpg', 'tiff'
        openslide_level: Resolution level for OpenSlide images (0=highest)
        format_filter: List of formats to process
        otsu_min_ratio: Minimum tissue ratio in Otsu mask (default 0.01 = 1%).
                       Use 0.01-0.02 for normal quality, 0.005-0.01 for low-quality staining.
        num_workers: Number of parallel workers (None = auto-calculate based on RAM)
        annotations_dir: Directory containing GeoJSON annotation files (optional, enables mask export)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if format_filter is None:
        allowed_formats = ALL_FORMATS
    else:
        allowed_formats = {
            ('.' + fmt.lstrip('.').lower()) if not fmt.startswith('.') else fmt.lower()
            for fmt in format_filter
        }

    archivos = os.listdir(input_dir)
    imagenes_a_procesar = [
        f for f in archivos
        if os.path.splitext(f)[1].lower() in allowed_formats
    ]

    if not imagenes_a_procesar:
        logger.info(f"No images found in: {input_dir}")
        return

    logger.info(f"Found {len(imagenes_a_procesar)} image(s) to process")
    logger.info(f"Target MPP: {target_mpp} µm/px")
    logger.info(f"Output format: {output_format.upper()}")

    # Sort images by file size (descending) for load balancing
    image_paths = [os.path.join(input_dir, f) for f in imagenes_a_procesar]
    image_paths.sort(key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0, reverse=True)
    imagenes_a_procesar = [os.path.basename(p) for p in image_paths]

    # Calculate num_workers if not provided
    if num_workers is None:
        num_workers = _safe_worker_count(image_paths)

    # Build task list for multiprocessing
    tasks = [
        (file_name, input_dir, output_dir, tile_size, overlap,
         target_mpp, zoom_scale, bg_threshold, output_format,
         openslide_level, otsu_min_ratio, annotations_dir)
        for file_name in imagenes_a_procesar
    ]

    logger.info(f"Starting parallel processing with {num_workers} workers...")
    start_time = time.time()

    # Process images in parallel
    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_single_image, tasks, chunksize=1), 1):
            if result:
                logger.info(f"Image {i}/{len(tasks)} completed: {result}")
            else:
                logger.warning(f"Image {i}/{len(tasks)} failed")

    elapsed = time.time() - start_time
    logger.info(f"Complete! Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    import sys

    input_dir = r"./Entradas"
    output_dir = r"./Salidas"
    annotations_dir = r"./Entradas"  # Set to None to disable mask export
    split_json = r"./Salidas/biopsy_split.json"

    # Load test set from shared split
    test_biopsies = None
    if Path(split_json).exists():
        try:
            with open(split_json, 'r') as f:
                split_data = json.load(f)
                test_biopsies = set(split_data.get('test', []))
                logger.info(f"Loaded test set from {split_json}: {len(test_biopsies)} biopsies")
                logger.info(f"Test biopsies: {sorted(test_biopsies)}")
        except Exception as e:
            logger.warning(f"Could not load split from {split_json}: {e}")
    else:
        logger.warning(f"Split file not found at {split_json}")
        logger.warning("Will process ALL images. Provide a valid split JSON to process only test set.")

    # Filter input directory to only include test biopsies
    if test_biopsies:
        # Create temporary directory with only test images
        temp_dir = Path(output_dir) / "_temp_test_input"
        temp_dir.mkdir(parents=True, exist_ok=True)

        input_path = Path(input_dir)
        processed_count = 0

        for item in input_path.iterdir():
            if item.is_file() and item.suffix.lower() in {'.svs', '.ndpi', '.vms', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}:
                biopsy_name = item.stem
                if biopsy_name in test_biopsies:
                    import shutil
                    shutil.copy2(item, temp_dir / item.name)
                    processed_count += 1
                    logger.info(f"Copied {item.name} to temp directory")

        logger.info(f"Filtered input: {processed_count} test images in {temp_dir}")
        input_dir = str(temp_dir)

    process_folder_to_subfolders(
        input_dir,
        output_dir,
        tile_size=1024,
        overlap=256,           # 25% overlap for better tissue coverage
        target_mpp=0.5,        # Standard resolution for OpenSlide (20x equivalent)
        zoom_scale=0.5,        # Downsampling for PIL (TIFF) — captures 4x more native pixels
        bg_threshold=15.0,     # Less aggressive filter for low-quality staining
        output_format='png',
        openslide_level=0,
        otsu_min_ratio=0.01,   # 1% minimum tissue (permissive for weak staining)
        annotations_dir=annotations_dir  # Enable mask export from GeoJSON
    )

    # Clean up temp directory
    if test_biopsies:
        import shutil
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up temp directory: {e}")
