#!/usr/bin/env python3
"""
Convert all image formats to pyramidal TIFF using libvips.

Supports: .tif, .png, .jpg, .jpeg, .svs, .ndpi, .vms, .jp2
Output: Pyramidal TIFF files with multiple resolution levels.
"""

import subprocess
import os
import sys
from pathlib import Path


SUPPORTED_FORMATS = {'.tif', '.png', '.jpg', '.jpeg', '.svs', '.ndpi', '.vms', '.jp2'}
SKIP_EXTENSIONS = {'.tiff', '.TIFF'}


def check_vips_installed():
    """Check if vips command is available."""
    try:
        subprocess.run(['vips', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def print_vips_installation_instructions():
    """Print instructions for installing libvips."""
    print("\n" + "=" * 70)
    print("ERROR: 'vips' command not found")
    print("=" * 70)
    print("\nTo install libvips on macOS:")
    print("  brew install libvips")
    print("\nAfter installation, verify with:")
    print("  vips --version")
    print("\n" + "=" * 70 + "\n")


def convert_to_pyramidal_tiff(input_path, output_path):
    """
    Convert a single image to pyramidal TIFF using vips.

    Args:
        input_path: Path to input image
        output_path: Path to output TIFF

    Returns:
        (success: bool, error_msg: str or None)
    """
    try:
        cmd = [
            'vips', 'tiffsave',
            str(input_path),
            str(output_path),
            '--tile',
            '--pyramid',
            '--compression=jpeg'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout per image
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, error_msg

        return True, None

    except subprocess.TimeoutExpired:
        return False, "Timeout (image too large or processing took >5min)"
    except Exception as e:
        return False, str(e)


def should_convert(file_path):
    """Check if file should be converted based on extension."""
    suffix = file_path.suffix.lower()
    return suffix in SUPPORTED_FORMATS


def get_output_path(file_path):
    """Generate output TIFF path."""
    return file_path.with_suffix('.tiff')


def convert_all_to_pyramidal_tiff(input_dir, verify=True):
    """
    Convert all supported image formats to pyramidal TIFF.

    Args:
        input_dir: Directory containing images
        verify: If True, ask for confirmation before deleting original files
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Directory not found: {input_dir}")
        return

    if not input_path.is_dir():
        print(f"Error: Not a directory: {input_dir}")
        return

    # Check vips availability
    if not check_vips_installed():
        print_vips_installation_instructions()
        return

    # Find all supported image files
    image_files = []
    for suffix in SUPPORTED_FORMATS:
        image_files.extend(input_path.glob(f'*{suffix}'))
        image_files.extend(input_path.glob(f'*{suffix.upper()}'))

    # Remove duplicates and sort
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No supported image files found in: {input_dir}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
        return

    print(f"\nFound {len(image_files)} image(s) to process")
    print(f"Output directory: {input_path}\n")

    stats = {
        'total': len(image_files),
        'converted': 0,
        'skipped': 0,
        'failed': 0,
        'failed_files': []
    }

    for file_path in image_files:
        # Skip if already TIFF
        if file_path.suffix.lower() in SKIP_EXTENSIONS:
            print(f"⏭️  Skipped (already .tiff): {file_path.name}")
            stats['skipped'] += 1
            continue

        output_path = get_output_path(file_path)
        print(f"Processing: {file_path.name} → {output_path.name}", end=" ")

        success, error_msg = convert_to_pyramidal_tiff(file_path, output_path)

        if success:
            # Conversion succeeded, delete original if verification passes
            should_delete = True

            if verify:
                response = input(f"\nDelete original? (y/n): ").strip().lower()
                should_delete = response == 'y'

            if should_delete:
                try:
                    file_path.unlink()
                    print("✅ Converted and deleted")
                    stats['converted'] += 1
                except Exception as e:
                    print(f"✅ Converted but failed to delete: {e}")
                    stats['converted'] += 1
            else:
                print("✅ Converted (original kept)")
                stats['converted'] += 1
        else:
            print(f"\n❌ Conversion failed: {error_msg}")
            stats['failed'] += 1
            stats['failed_files'].append((file_path.name, error_msg))

    # Print summary
    print("\n" + "=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Total files:     {stats['total']}")
    print(f"Converted:       {stats['converted']}")
    print(f"Skipped:         {stats['skipped']}")
    print(f"Failed:          {stats['failed']}")

    if stats['failed_files']:
        print("\nFailed files:")
        for filename, error in stats['failed_files']:
            print(f"  - {filename}: {error}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    input_dir = r"/Users/olivera/Documents/Proyecto_Final_Glomerulos/Entradas"
    convert_all_to_pyramidal_tiff(input_dir, verify=False)
