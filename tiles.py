import os
import json
import logging
from PIL import Image
import numpy as np
from pathlib import Path

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

OPENSLIDE_FORMATS = {'.svs', '.ndpi', '.vms'}
PIL_FORMATS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
ALL_FORMATS = OPENSLIDE_FORMATS | PIL_FORMATS


def detect_format(file_path):
    """Detect image format by file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in OPENSLIDE_FORMATS:
        return 'openslide'
    elif ext in PIL_FORMATS:
        return 'pil'
    return None


def generate_otsu_mask(img_or_slide, downsample_factor=20):
    """Generate binary tissue mask via Otsu thresholding on HSV saturation."""
    if not OPENCV_AVAILABLE:
        return None, None

    if hasattr(img_or_slide, 'get_thumbnail'):  # OpenSlide
        w, h = img_or_slide.level_dimensions[0]
        thumb = img_or_slide.get_thumbnail((w // downsample_factor, h // downsample_factor))
    else:  # PIL Image
        w, h = img_or_slide.size
        thumb = img_or_slide.resize((w // downsample_factor, h // downsample_factor), Image.Resampling.LANCZOS)

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


def process_folder_to_subfolders(input_dir, output_dir, tile_size=1536, overlap=512,
                                target_mpp=0.5, zoom_scale=0.5, bg_threshold=15.0, output_format='png',
                                openslide_level=0, format_filter=None, otsu_min_ratio=0.02):
    """
    Process histopathology images into tiles using MPP-aware resolution.

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

    for file_name in imagenes_a_procesar:
        image_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(file_name)[1].lower()

        image_output_dir = os.path.join(output_dir, "Imagen", base_name)
        Path(image_output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing: {file_name}")
        logger.info(f"  Type: {file_ext[1:].upper()}")
        logger.info(f"  Output dir: {image_output_dir}")

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

                # Get native MPP and compute downsampling
                try:
                    native_mpp = float(slide.properties.get('openslide.mpp-x', 0))
                    if native_mpp > 0 and target_mpp > 0:
                        downsample_ratio = target_mpp / native_mpp
                        logger.info(f"  Native MPP: {native_mpp:.4f} µm/px")
                        logger.info(f"  Downsampling ratio: {downsample_ratio:.4f}")
                    else:
                        raise ValueError("Invalid MPP values")
                except (ValueError, TypeError, AttributeError):
                    logger.warning("Could not determine native MPP, using full resolution")
                    native_mpp = None
                    downsample_ratio = 1.0

                # Generate Otsu mask
                if OPENCV_AVAILABLE:
                    mask, downsample_factor = generate_otsu_mask(slide, downsample_factor=20)
                    logger.info(f"  Otsu mask generated (downsample × {downsample_factor})")

                # Compute iteration parameters in native space
                native_tile_size = int(tile_size * downsample_ratio)
                native_stride = int((tile_size - overlap) * downsample_ratio)

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
                        saved_count += 1

                        if saved_count % 100 == 0:
                            logger.info(f"  Saved {saved_count} tiles...")

                logger.info(f"  Final: {saved_count} tiles saved, {skipped_count} background tiles skipped")

                # Save metadata
                metadata = {
                    "native_mpp": native_mpp,
                    "target_mpp": target_mpp,
                    "downsample_ratio": downsample_ratio,
                    "tile_size": tile_size,
                    "overlap": overlap,
                    "native_width": native_width,
                    "native_height": native_height,
                    "format": "openslide"
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
                    "zoom_scale": zoom_scale
                }

            else:
                raise ValueError(f"Unsupported format: {file_ext}")

            # Save metadata.json
            metadata_path = os.path.join(image_output_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"  Metadata saved to metadata.json")

        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}", exc_info=True)

    logger.info("Complete!")


if __name__ == "__main__":
    input_dir = r"/Users/olivera/Documents/Proyecto_Final_Glomerulos/Entradas"
    output_dir = r"/Users/olivera/Documents/Proyecto_Final_Glomerulos/Salidas"

    process_folder_to_subfolders(
        input_dir,
        output_dir,
        tile_size=1536,
        overlap=512,           # 50% overlap for better tissue coverage
        target_mpp=0.5,        # Standard resolution for OpenSlide (20x equivalent)
        zoom_scale=0.5,        # Downsampling for PIL (TIFF) — captures 4x more native pixels
        bg_threshold=15.0,     # Less aggressive filter for low-quality staining
        output_format='png',
        openslide_level=0,
        otsu_min_ratio=0.02    # 1% minimum tissue (permissive for weak staining)
    )
