#!/usr/bin/env python3
"""Reconstruct WSI from annotated tiles (tiling_anotaciones.py output)."""

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from typing import Optional, Tuple, Dict, Any
import os

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Legend drawing constants
LEGEND_BOX_SIZE = 40
LEGEND_ITEM_SPACING = 12
LEGEND_WIDTH = 320
LEGEND_MARGIN = 20

# Memory estimation constants
DEFAULT_SLIDE_MEMORY_ESTIMATE = 500 * 1024**2  # 500 MB default


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color '#RRGGBB' to (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def calculate_scaled_tile_dimensions(tile_width: int, tile_height: int, zoom_scale: float, recon_scale: int) -> Tuple[int, int]:
    """Calculate scaled tile dimensions accounting for zoom_scale and recon_scale."""
    tile_native_width = int(tile_width / zoom_scale)
    tile_native_height = int(tile_height / zoom_scale)
    tile_w_scaled = max(1, tile_native_width // recon_scale)
    tile_h_scaled = max(1, tile_native_height // recon_scale)
    return tile_w_scaled, tile_h_scaled


def draw_legend(
    image: Image.Image,
    class_map: Dict[str, int],
    class_colors: Dict[str, str],
) -> None:
    """Draw legend showing class colors and names on image."""
    draw = ImageDraw.Draw(image, 'RGBA')

    # Sort by class ID to ensure consistent order
    sorted_classes = sorted(class_map.items(), key=lambda item: item[1])
    non_bg_classes = [c for c in sorted_classes if c[0] != 'background']

    if not non_bg_classes:
        return

    # Calculate legend dimensions
    legend_height = LEGEND_MARGIN + (len(non_bg_classes) * (LEGEND_BOX_SIZE + LEGEND_ITEM_SPACING)) + LEGEND_MARGIN

    # Draw semi-transparent black background for legend
    draw.rectangle(
        [LEGEND_MARGIN - 5, LEGEND_MARGIN - 5, LEGEND_MARGIN + LEGEND_WIDTH, LEGEND_MARGIN + legend_height],
        fill=(0, 0, 0, 220)
    )

    y = LEGEND_MARGIN
    x = LEGEND_MARGIN + 10

    for class_name, class_id in non_bg_classes:
        hex_color = class_colors.get(class_name, '#ffffff')
        r, g, b = hex_to_rgb(hex_color)

        # Draw colored rectangle
        draw.rectangle(
            [x, y, x + LEGEND_BOX_SIZE, y + LEGEND_BOX_SIZE],
            fill=(r, g, b, 255),
            outline='white'
        )

        # Draw class name text - larger and bold
        text = class_name.replace('_', ' ')
        text_x = x + LEGEND_BOX_SIZE + 15
        text_y = y + 8

        # Draw text with black outline for visibility
        for adj_x, adj_y in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            draw.text((text_x + adj_x, text_y + adj_y), text, fill='black')
        draw.text((text_x, text_y), text, fill='white')

        y += LEGEND_BOX_SIZE + LEGEND_ITEM_SPACING


class MemoryMonitor:
    """Monitor memory usage and adapt worker count dynamically."""

    def __init__(self, mem_percent: int = 80):
        self.mem_percent = mem_percent
        if HAS_PSUTIL:
            try:
                self.total_memory = psutil.virtual_memory().total
            except (AttributeError, OSError):
                self.total_memory = 16 * 1024**3
        else:
            self.total_memory = 16 * 1024**3
        self.max_allowed = (self.total_memory * mem_percent) // 100
        self.slide_memory_cache = {}

    def get_available_memory(self) -> int:
        """Return available memory in bytes."""
        if not HAS_PSUTIL:
            return self.total_memory // 2
        return psutil.virtual_memory().available

    def get_memory_usage(self) -> int:
        """Return current process memory usage in bytes."""
        if not HAS_PSUTIL:
            return 0
        return psutil.Process(os.getpid()).memory_info().rss

    def can_load_slide(self, slide_size_estimate: int) -> bool:
        """Check if we can load another slide without exceeding mem_percent."""
        current_usage = self.get_memory_usage()
        available = self.get_available_memory()
        return (current_usage + slide_size_estimate) <= self.max_allowed

    def get_recommended_workers(self) -> int:
        """Estimate workers without exceeding mem_percent."""
        if not HAS_PSUTIL:
            return 2

        avg_mem = (
            sum(self.slide_memory_cache.values()) /
            len(self.slide_memory_cache)
            if self.slide_memory_cache
            else DEFAULT_SLIDE_MEMORY_ESTIMATE
        )
        max_workers = max(1, (self.max_allowed // int(avg_mem)))
        available = self.get_available_memory()
        return int(
            min(
                max(1, available // int(avg_mem * 1.2)),
                max_workers
            )
        )

    def report_slide_memory(self, slide_name: str, memory_used: int):
        """Cache memory usage of a slide for future estimates."""
        self.slide_memory_cache[slide_name] = memory_used


def load_annotations(annotations_path: Path) -> Dict[str, Any]:
    """Load and parse annotations.json file."""
    with open(annotations_path, 'r') as f:
        return json.load(f)


def reconstruct_slide(
    slide_name: str,
    input_dir: Path = Path('Salidas/Tiles_UNet'),
    output_dir: Optional[Path] = None,
    recon_scale: int = 10,
    overlay_alpha: int = 180,
) -> Tuple[bool, str, int]:
    """Reconstruct a single slide by pasting tiles at their native coordinates with colored overlays."""
    if output_dir is None:
        output_dir = input_dir

    mem_start = 0
    if HAS_PSUTIL:
        try:
            mem_start = psutil.Process(os.getpid()).memory_info().rss
        except (AttributeError, OSError):
            mem_start = 0

    slide_dir = input_dir / slide_name
    annotations_path = slide_dir / 'annotations.json'

    if not annotations_path.exists():
        return False, f"annotations.json not found", 0

    try:
        annotations = load_annotations(annotations_path)
    except (json.JSONDecodeError, IOError) as e:
        return False, f"Failed to load annotations: {e}", 0

    native_width = annotations.get('width_native', 0)
    native_height = annotations.get('height_native', 0)
    tiles = annotations.get('tiles', [])
    class_map = annotations.get('class_map', {})
    class_colors = annotations.get('class_colors', {})
    tiling_config = annotations.get('tiling_config', {})
    zoom_scale = tiling_config.get('zoom_scale', 1.0)

    if native_width == 0 or native_height == 0:
        return False, "Invalid native dimensions", 0

    canvas_width = max(1, native_width // recon_scale)
    canvas_height = max(1, native_height // recon_scale)

    wsi_img = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
    mask_canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))

    for tile_data in tiles:
        image_path = slide_dir / tile_data.get('image', '')
        mask_path = slide_dir / tile_data.get('mask', '')
        origin_native = tile_data.get('origin_native', [0, 0])

        scaled_x = origin_native[0] // recon_scale
        scaled_y = origin_native[1] // recon_scale
        scaled_origin = (scaled_x, scaled_y)

        if image_path.exists():
            try:
                tile_img = Image.open(image_path).convert('RGB')
                w_s, h_s = calculate_scaled_tile_dimensions(
                    tile_img.width, tile_img.height, zoom_scale, recon_scale
                )
                tile_img_scaled = tile_img.resize(
                    (w_s, h_s), Image.LANCZOS
                )
                wsi_img.paste(tile_img_scaled, scaled_origin)
            except (IOError, OSError) as e:
                msg = f"  ⚠️  Warning: Failed to load image {image_path}: {e}"
                print(msg)

        if mask_path.exists():
            try:
                tile_mask = Image.open(mask_path).convert('RGB')
                w_s, h_s = calculate_scaled_tile_dimensions(
                    tile_mask.width, tile_mask.height, zoom_scale, recon_scale
                )
                tile_mask_scaled = tile_mask.resize(
                    (w_s, h_s), Image.NEAREST
                )

                # Make black background pixels transparent
                tile_mask_rgba = tile_mask_scaled.convert('RGBA')
                mask_arr = np.array(tile_mask_rgba)
                is_black = (
                    (mask_arr[:, :, 0] == 0) &
                    (mask_arr[:, :, 1] == 0) &
                    (mask_arr[:, :, 2] == 0)
                )
                alpha_arr = np.where(is_black, 0, overlay_alpha)
                tile_mask_rgba.putalpha(
                    Image.fromarray(alpha_arr.astype(np.uint8))
                )

                # Composite tile onto main mask canvas
                tile_canvas = Image.new(
                    'RGBA', (canvas_width, canvas_height), (0, 0, 0, 0)
                )
                tile_canvas.paste(
                    tile_mask_rgba, scaled_origin, tile_mask_rgba
                )
                mask_canvas = Image.alpha_composite(
                    mask_canvas, tile_canvas
                )

            except (IOError, OSError) as e:
                msg = f"  ⚠️  Warning: Failed to load mask {mask_path}: {e}"
                print(msg)

    output_recon_dir = output_dir / slide_name / 'reconstructions'
    output_recon_dir.mkdir(parents=True, exist_ok=True)

    try:
        recon_path = (
            output_recon_dir / f"{slide_name}_reconstruction.png"
        )
        mask_colored_path = (
            output_recon_dir / f"{slide_name}_mask_colored.png"
        )
        overlay_path = output_recon_dir / f"{slide_name}_overlay.png"

        wsi_img.save(recon_path, 'PNG')

        # Save colored mask
        mask_rgb_out = mask_canvas.convert('RGB')
        mask_rgb_out.save(mask_colored_path, 'PNG')

        # Create colored overlay
        wsi_rgba = wsi_img.convert('RGBA')
        overlay_canvas = Image.alpha_composite(wsi_rgba, mask_canvas)
        overlay_rgb = overlay_canvas.convert('RGB')

        # Draw legend if class info available
        if class_map and class_colors:
            draw_legend(overlay_rgb, class_map, class_colors)

        overlay_rgb.save(overlay_path, 'PNG')

        mem_end = 0
        if HAS_PSUTIL:
            try:
                mem_end = psutil.Process(os.getpid()).memory_info().rss
            except (AttributeError, OSError):
                mem_end = mem_start
        mem_used = mem_end - mem_start

        msg = (
            f"{slide_name}: 3 outputs saved "
            "(reconstruction + colored_mask + overlay)"
        )
        return True, msg, mem_used

    except (IOError, OSError) as e:
        return False, f"Failed to save outputs: {e}", 0


def process_all_slides_dynamic(
    input_dir: Path,
    output_dir: Path,
    recon_scale: int,
    mem_monitor: MemoryMonitor,
    overlay_alpha: int = 180,
):
    """Process all slides with dynamic worker adaptation."""
    slides = sorted([d.name for d in input_dir.iterdir() if d.is_dir()])
    if not slides:
        print(f"No slides found in {input_dir}")
        return

    print(f"📊 Dynamic mode: Using up to {mem_monitor.mem_percent}% of {mem_monitor.total_memory // 1024**3} GB RAM")
    print(f"Processing {len(slides)} slides with adaptive parallelism...\n")

    completed = 0
    failed = 0
    pbar = tqdm(total=len(slides), desc="Reconstructing", unit="slide")

    max_workers = int(min(8, mem_monitor.get_recommended_workers() * 2))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for slide in slides[:3]:
            future = executor.submit(reconstruct_slide, slide, input_dir, output_dir, recon_scale, overlay_alpha)
            futures[future] = slide

        slide_idx = 3

        while futures:
            done, _ = wait(futures, timeout=0.5, return_when='FIRST_COMPLETED')

            for future in done:
                slide_name = futures.pop(future)
                success, message, mem_used = future.result()

                if success:
                    mem_monitor.report_slide_memory(slide_name, mem_used)
                    print(f"  ✓ {message} ({mem_used // 1024**2} MB)")
                    completed += 1
                else:
                    print(f"  ❌ {message}")
                    failed += 1

                pbar.update(1)

            # Calculate average memory once per completed batch
            avg_mem = (
                sum(mem_monitor.slide_memory_cache.values()) /
                len(mem_monitor.slide_memory_cache)
                if mem_monitor.slide_memory_cache
                else DEFAULT_SLIDE_MEMORY_ESTIMATE
            )

            while slide_idx < len(slides):
                if mem_monitor.can_load_slide(int(avg_mem)):
                    next_slide = slides[slide_idx]
                    future = executor.submit(
                        reconstruct_slide,
                        next_slide,
                        input_dir,
                        output_dir,
                        recon_scale,
                        overlay_alpha
                    )
                    futures[future] = next_slide
                    slide_idx += 1
                else:
                    break

    pbar.close()
    print(f"\n✅ Complete: {completed} OK, {failed} failed")


def process_all_slides_fixed(
    input_dir: Path,
    output_dir: Path,
    recon_scale: int,
    num_workers: int,
    overlay_alpha: int = 180,
):
    """Process all slides with fixed number of workers."""
    slides = sorted([d.name for d in input_dir.iterdir() if d.is_dir()])
    if not slides:
        print(f"No slides found in {input_dir}")
        return

    print(f"Processing {len(slides)} slides with {num_workers} fixed workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(reconstruct_slide, slide, input_dir, output_dir, recon_scale, overlay_alpha)
            for slide in slides
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reconstructing", unit="slide"):
            success, message, mem_used = future.result()
            if success:
                print(f"  ✓ {message}")
            else:
                print(f"  ❌ {message}")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct WSI from annotated tiles (dynamic or fixed workers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # All slides, adaptive parallelism (default, respects 70% RAM limit)
  python reconstruct_slide.py

  # Single slide
  python reconstruct_slide.py BR-129-PAS-24-CONV

  # All slides, fixed 4 workers
  python reconstruct_slide.py --workers 4

  # All slides, dynamic but limit to 50% RAM
  python reconstruct_slide.py --mem-percent 50
        """,
    )
    parser.add_argument(
        'slide_name',
        nargs='?',
        help='Slide name (e.g., BR-129-PAS-24-CONV); omit with --all',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('Salidas/Tiles_UNet'),
        help='Input directory with {slide_name}/ subdirs (default: Salidas/Tiles_UNet)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: same as input-dir)',
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=10,
        help='Downsample factor (default: 10)',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all slides in input-dir',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Fixed number of workers; if omitted, use dynamic adaption',
    )
    parser.add_argument(
        '--mem-percent',
        type=int,
        default=50,
        help='Max %% of system RAM to use in dynamic mode (default: 50)',
    )
    parser.add_argument(
        '--overlay-alpha',
        type=int,
        default=180,
        help='Alpha opacity of class colors in overlay (0-255, default: 180)',
    )

    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir

    if args.all or not args.slide_name:
        if args.workers is not None:
            process_all_slides_fixed(args.input_dir, output_dir, args.scale, args.workers, args.overlay_alpha)
        else:
            if not HAS_PSUTIL:
                print("⚠️  psutil not installed. Installing it would enable dynamic memory monitoring.")
                print("   Falling back to fixed 2 workers.")
                process_all_slides_fixed(args.input_dir, output_dir, args.scale, 2, args.overlay_alpha)
            else:
                mem_monitor = MemoryMonitor(args.mem_percent)
                process_all_slides_dynamic(args.input_dir, output_dir, args.scale, mem_monitor, args.overlay_alpha)
    else:
        if not args.slide_name:
            parser.print_help()
            return

        print(f"Reconstructing {args.slide_name}...")
        success, message, mem_used = reconstruct_slide(args.slide_name, args.input_dir, output_dir, args.scale, args.overlay_alpha)
        if success:
            print(f"  ✓ {message}")
        else:
            print(f"  ❌ {message}")


if __name__ == '__main__':
    main()
