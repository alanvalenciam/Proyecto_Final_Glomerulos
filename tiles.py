import os
from PIL import Image
import numpy as np
from pathlib import Path

# Remove pixel limit for large images
Image.MAX_IMAGE_PIXELS = None

# OpenSlide support for SVS, NDPI, VMS formats
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("WARNING: openslide-python not installed.")
    print("Install with: pip install openslide-python")
    print("SVS, NDPI, and VMS formats will not be supported.\n")

# Supported formats
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


def read_openslide_image(file_path, level=0):
    """Read SVS/NDPI/VMS file using OpenSlide at specified resolution level."""
    if not OPENSLIDE_AVAILABLE:
        raise RuntimeError(
            f"OpenSlide not available. Install with: pip install openslide-python"
        )

    slide = openslide.open_slide(file_path)
    width, height = slide.level_dimensions[level]
    level_count = slide.level_count

    # Read image at specified level
    pil_image = slide.read_region((0, 0), level, (width, height))

    # Convert to RGB
    if pil_image.mode == 'RGBA':
        rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
        rgb_image.paste(pil_image, mask=pil_image.split()[3])
        pil_image = rgb_image
    elif pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    return pil_image, width, height, level_count


def read_pil_image(file_path):
    """Read TIFF, PNG, JPEG using PIL."""
    img = Image.open(file_path)

    if img.mode == 'RGBA':
        rgb_image = Image.new('RGB', img.size, (255, 255, 255))
        rgb_image.paste(img, mask=img.split()[3])
        img = rgb_image
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    return img


def save_tile(tile, tile_path, output_format='png'):
    """Save tile in specified format (png, jpg, tiff)."""
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
            print(f"    PNG save failed, falling back to JPEG")
            tile_path = tile_path.replace('.png', '.jpg')
            if tile.mode != 'RGB':
                tile = tile.convert('RGB')
            tile.save(tile_path, 'JPEG', quality=95)
        else:
            raise


def is_mostly_background(tile, min_tissue_ratio=0.05):
    """Return True if tile lacks tissue (should be skipped).

    Uses HSV saturation: PAS tissue (purple/magenta) has S > 80,
    scanner border/glass background has S clustered at 28-40.

    Args:
        tile: PIL Image to check
        min_tissue_ratio: fraction of pixels with S > 80 required to keep tile (default 0.05 = 5%)

    Returns:
        True if tile is background only (should discard)
        False if tile contains tissue (should keep)
    """
    hsv = np.array(tile.convert('HSV'))
    saturation = hsv[:, :, 1]  # Extract S channel (0-255)

    # Count pixels with high saturation (tissue has S > 80, background has S < 40)
    tissue_pixels = np.sum(saturation > 80)
    tissue_ratio = tissue_pixels / saturation.size

    # Discard tile if less than min_tissue_ratio of pixels are tissue-colored
    return tissue_ratio < min_tissue_ratio


def process_folder_to_subfolders(input_dir, output_dir, tile_size=1536, overlap=256,
                                zoom_scale=0.5, bg_threshold=0.05, output_format='png',
                                openslide_level=0, format_filter=None):
    """
    Process histopathology images into tiles.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output tiles
        tile_size: Size of each tile in pixels (default 1536)
        overlap: Overlap between tiles in pixels (default 0)
        zoom_scale: Scaling factor for image before tiling (default 1.0, e.g. 2.0 = 2x)
        bg_threshold: Min tissue ratio (0-1) to keep tile. Default 0.05 means keep tile only if ≥5% pixels are tissue-colored (S>80 in HSV)
        output_format: Output format: 'png', 'jpg', 'tiff' (default 'png')
        openslide_level: Resolution level for OpenSlide images (0=highest)
        format_filter: List of formats to process (e.g. ['tiff', 'svs'])
                      If None, process all supported formats
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine which formats to process
    if format_filter is None:
        allowed_formats = ALL_FORMATS
    else:
        allowed_formats = {
            ('.' + fmt.lstrip('.').lower()) if not fmt.startswith('.') else fmt.lower()
            for fmt in format_filter
        }

    # Find images
    archivos = os.listdir(input_dir)
    imagenes_a_procesar = [
        f for f in archivos
        if os.path.splitext(f)[1].lower() in allowed_formats
    ]

    if not imagenes_a_procesar:
        print(f"No images found in: {input_dir}")
        return

    print(f"Found {len(imagenes_a_procesar)} image(s) to process")
    print(f"Output format: {output_format.upper()}\n")

    # Process each image
    for file_name in imagenes_a_procesar:
        image_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        file_ext = os.path.splitext(file_name)[1].lower()

        image_output_dir = os.path.join(output_dir, "Imagen", base_name)
        Path(image_output_dir).mkdir(parents=True, exist_ok=True)

        print(f"Processing: {file_name}")
        print(f"  Type: {file_ext[1:].upper()}")
        print(f"  Output dir: {image_output_dir}")

        try:
            handler = detect_format(image_path)

            if handler == 'openslide':
                img, width, height, level_count = read_openslide_image(
                    image_path, level=openslide_level
                )
                print(f"  Dimensions (level {openslide_level}): {width}x{height}")
                print(f"  Resolution levels: {level_count}")
            elif handler == 'pil':
                img = read_pil_image(image_path)
                width, height = img.size
                print(f"  Dimensions: {width}x{height}")
            else:
                raise ValueError(f"Unsupported format: {file_ext}")

            # Apply zoom scaling
            if zoom_scale != 1.0:
                new_width = int(width * zoom_scale)
                new_height = int(height * zoom_scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                width, height = new_width, new_height
                print(f"  Scaled to {zoom_scale}x: {width}x{height}")

            stride = tile_size - overlap
            saved_count = 0

            # Extract and save tiles
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    x_end = min(x + tile_size, width)
                    y_end = min(y + tile_size, height)

                    tile = img.crop((x, y, x_end, y_end))

                    # Skip background tiles
                    if not is_mostly_background(tile, bg_threshold):
                        # Calculate original image coordinates (convert from scaled to original)
                        orig_x = int(x / zoom_scale)
                        orig_y = int(y / zoom_scale)
                        orig_x_end = int(x_end / zoom_scale)
                        orig_y_end = int(y_end / zoom_scale)

                        ext = '.' + output_format.lstrip('.')
                        tile_name = f"{base_name}_tile_x{orig_x:05d}_y{orig_y:05d}_endx{orig_x_end:05d}_endy{orig_y_end:05d}{ext}"
                        tile_path = os.path.join(image_output_dir, tile_name)

                        # Pad tile to fixed size (1536x1536) with white background
                        if tile.size != (tile_size, tile_size):
                            padded_tile = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
                            padded_tile.paste(tile, (0, 0))
                            tile = padded_tile

                        # Save tile
                        save_tile(tile, tile_path, output_format)

                        saved_count += 1

                        if saved_count % 100 == 0:
                            print(f"  Saved {saved_count} tiles...", end='\r')

        except Exception as e:
            print(f"\n  Error processing {file_name}: {e}\n")
            import traceback
            traceback.print_exc()

    print("Complete!")


if __name__ == "__main__":
    input_dir = r"/Users/olivera/Documents/Proyecto_Final_Glomerulos/Entradas"
    output_dir = r"/Users/olivera/Documents/Proyecto_Final_Glomerulos/Salidas"

    process_folder_to_subfolders(
        input_dir,
        output_dir,
        tile_size=1536,
        overlap=256,
        zoom_scale=0.5,
        bg_threshold=0.05,
        output_format='png',
        openslide_level=0
    )