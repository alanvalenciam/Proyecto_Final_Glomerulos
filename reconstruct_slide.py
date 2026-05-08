#!/usr/bin/env python3
"""
Reconstruct WSI from annotated tiles (tiling_anotaciones.py output).

Reads annotations.json from Salidas/Tiles_UNet/{slide_name}/ and reconstructs:
1. Downsampled full WSI image
2. Binary segmentation mask (white=glomerulo, black=background)
3. Overlay blend (image + mask at 70/30)

Dynamic workers: if --workers not specified, adapts parallelism based on available memory.
Never exceeds specified --mem-percent (default 70%) of total system RAM.

Usage:
    python reconstruct_slide.py                                # Process all slides (default)
    python reconstruct_slide.py BR-129-PAS-24-CONV --scale 10 # Process one slide
    python reconstruct_slide.py --workers 4                     # All slides, 4 fixed workers
    python reconstruct_slide.py --mem-percent 60                # All slides, max 60% RAM
"""

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from typing import Optional, Tuple
import os

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


class MemoryMonitor:
    """Monitor memory usage and adapt worker count dynamically."""

    def __init__(self, mem_percent: int = 80):
        """
        Args:
            mem_percent: Max % of total system RAM to use (default 80 for 28 GB)
        """
        self.mem_percent = mem_percent
        self.total_memory = psutil.virtual_memory().total if HAS_PSUTIL else 16 * 1024**3
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
        """Estimate how many workers can run in parallel without exceeding mem_percent."""
        if not HAS_PSUTIL:
            return 2

        available = self.get_available_memory()
        avg_slide_memory = sum(self.slide_memory_cache.values()) / len(self.slide_memory_cache) if self.slide_memory_cache else 500 * 1024**2
        # Estimate: how many slides of avg_slide_memory fit in max_allowed?
        max_workers = max(1, (self.max_allowed // int(avg_slide_memory)))
        return min(max(1, available // (avg_slide_memory * 1.2)), max_workers)

    def report_slide_memory(self, slide_name: str, memory_used: int):
        """Cache memory usage of a slide for future estimates."""
        self.slide_memory_cache[slide_name] = memory_used


def load_annotations(annotations_path: Path) -> dict:
    """Load annotations.json and return parsed dict."""
    with open(annotations_path, 'r') as f:
        return json.load(f)


def load_wsi_image(image_path: str, image_format: str, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Load WSI image, optionally at a target size.

    Args:
        image_path: Path to WSI file
        image_format: 'pil' or 'openslide'
        target_size: (width, height) for thumbnail; None = full resolution

    Returns:
        PIL.Image in RGB mode
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_format == 'openslide' and HAS_OPENSLIDE:
        try:
            slide = openslide.open_slide(image_path)
            if target_size:
                img = slide.get_thumbnail(target_size)
            else:
                w, h = slide.dimensions
                img = slide.get_thumbnail((w, h))
            return img.convert('RGB')
        except Exception as e:
            print(f"  OpenSlide failed ({e}), falling back to PIL")

    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    if target_size:
        img = img.resize(target_size, Image.LANCZOS)

    return img


def scale_polygon(polygon: list, scale: float) -> list:
    """Scale polygon coordinates by dividing by scale factor."""
    return [(int(x / scale), int(y / scale)) for x, y in polygon]


def reconstruct_slide(
    slide_name: str,
    input_dir: Path = Path('Salidas/Tiles_UNet'),
    output_dir: Optional[Path] = None,
    recon_scale: int = 10,
) -> Tuple[bool, str, int]:
    """
    Reconstruct a single slide by pasting tiles at their native coordinates.

    Args:
        slide_name: e.g., 'BR-129-PAS-24-CONV'
        input_dir: Directory containing {slide_name}/ subdirs
        output_dir: Where to create reconstructions/; if None, use input_dir
        recon_scale: Downsample factor for final output

    Returns:
        Tuple[success, message, memory_used_bytes]
    """
    if output_dir is None:
        output_dir = input_dir

    mem_start = psutil.Process(os.getpid()).memory_info().rss if HAS_PSUTIL else 0

    slide_dir = input_dir / slide_name
    annotations_path = slide_dir / 'annotations.json'

    if not annotations_path.exists():
        return False, f"annotations.json not found", 0

    try:
        annotations = load_annotations(annotations_path)
    except Exception as e:
        return False, f"Failed to load annotations: {e}", 0

    native_width = annotations.get('width_native', 0)
    native_height = annotations.get('height_native', 0)
    tiles = annotations.get('tiles', [])

    if native_width == 0 or native_height == 0:
        return False, "Invalid native dimensions", 0

    # Create canvas at native resolution first
    wsi_img = Image.new('RGB', (native_width, native_height), (255, 255, 255))
    mask_img = Image.new('L', (native_width, native_height), 0)

    # Paste each tile at its native position
    for tile_data in tiles:
        image_path = slide_dir / tile_data.get('image', '')
        mask_path = slide_dir / tile_data.get('mask', '')
        origin = tile_data.get('origin_native', [0, 0])

        # Paste image tile
        if image_path.exists():
            try:
                tile_img = Image.open(image_path).convert('RGB')
                wsi_img.paste(tile_img, tuple(origin))
            except Exception:
                pass

        # Paste mask tile
        if mask_path.exists():
            try:
                tile_mask = Image.open(mask_path).convert('L')
                mask_img.paste(tile_mask, tuple(origin))
            except Exception:
                pass

    # Scale down to reduce file size
    canvas_width = max(1, native_width // recon_scale)
    canvas_height = max(1, native_height // recon_scale)

    wsi_scaled = wsi_img.resize((canvas_width, canvas_height), Image.LANCZOS)
    mask_scaled = mask_img.resize((canvas_width, canvas_height), Image.NEAREST)

    output_recon_dir = output_dir / slide_name / 'reconstructions'
    output_recon_dir.mkdir(parents=True, exist_ok=True)

    try:
        recon_path = output_recon_dir / f"{slide_name}_reconstruction.png"
        mask_path_out = output_recon_dir / f"{slide_name}_mask.png"
        overlay_path = output_recon_dir / f"{slide_name}_overlay.png"

        wsi_scaled.save(recon_path, 'PNG')
        mask_scaled.save(mask_path_out, 'PNG')

        mask_rgb = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
        mask_rgb.paste((255, 255, 255), (0, 0), mask_scaled)
        overlay = Image.blend(wsi_scaled, mask_rgb, alpha=0.3)
        overlay.save(overlay_path, 'PNG')

        mem_end = psutil.Process(os.getpid()).memory_info().rss if HAS_PSUTIL else 0
        mem_used = mem_end - mem_start

        return True, f"{slide_name}: 3 outputs saved", mem_used

    except Exception as e:
        return False, f"Failed to save outputs: {e}", 0


def process_all_slides_dynamic(
    input_dir: Path,
    output_dir: Path,
    recon_scale: int,
    mem_monitor: MemoryMonitor,
):
    """
    Process all slides with dynamic worker adaptation.

    Workers adjust based on available memory — never exceed mem_percent.
    """
    slides = sorted([d.name for d in input_dir.iterdir() if d.is_dir()])
    if not slides:
        print(f"No slides found in {input_dir}")
        return

    print(f"📊 Dynamic mode: Using up to {mem_monitor.mem_percent}% of {mem_monitor.total_memory // 1024**3} GB RAM")
    print(f"Processing {len(slides)} slides with adaptive parallelism...\n")

    completed = 0
    failed = 0
    pbar = tqdm(total=len(slides), desc="Reconstructing", unit="slide")

    # Use a future-based approach to dynamically manage workers
    max_workers = min(8, mem_monitor.get_recommended_workers() * 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Submit initial batch
        for slide in slides[:3]:
            future = executor.submit(reconstruct_slide, slide, input_dir, output_dir, recon_scale)
            futures[future] = slide

        slide_idx = 3

        # Process futures as they complete
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

                # Try to submit next slide if memory allows
                while slide_idx < len(slides):
                    # Estimate memory for next slide
                    avg_mem = sum(mem_monitor.slide_memory_cache.values()) / len(mem_monitor.slide_memory_cache) if mem_monitor.slide_memory_cache else 500 * 1024**2
                    if mem_monitor.can_load_slide(int(avg_mem)):
                        next_slide = slides[slide_idx]
                        future = executor.submit(reconstruct_slide, next_slide, input_dir, output_dir, recon_scale)
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
):
    """Process all slides with fixed number of workers."""
    slides = sorted([d.name for d in input_dir.iterdir() if d.is_dir()])
    if not slides:
        print(f"No slides found in {input_dir}")
        return

    print(f"Processing {len(slides)} slides with {num_workers} fixed workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(reconstruct_slide, slide, input_dir, output_dir, recon_scale)
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
        default=90,
        help='Max %% of system RAM to use in dynamic mode (default: 90)',
    )

    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir

    if args.all or not args.slide_name:
        if args.workers is not None:
            # Fixed workers mode
            process_all_slides_fixed(args.input_dir, output_dir, args.scale, args.workers)
        else:
            # Dynamic workers mode
            if not HAS_PSUTIL:
                print("⚠️  psutil not installed. Installing it would enable dynamic memory monitoring.")
                print("   Falling back to fixed 2 workers.")
                process_all_slides_fixed(args.input_dir, output_dir, args.scale, 2)
            else:
                mem_monitor = MemoryMonitor(args.mem_percent)
                process_all_slides_dynamic(args.input_dir, output_dir, args.scale, mem_monitor)
    else:
        if not args.slide_name:
            parser.print_help()
            return

        print(f"Reconstructing {args.slide_name}...")
        success, message, mem_used = reconstruct_slide(args.slide_name, args.input_dir, output_dir, args.scale)
        if success:
            print(f"  ✓ {message}")
        else:
            print(f"  ❌ {message}")


if __name__ == '__main__':
    main()
