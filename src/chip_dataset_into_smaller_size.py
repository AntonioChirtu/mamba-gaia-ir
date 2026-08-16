import os
from PIL import Image
from pathlib import Path

# 1. Increase PIL's decompression bomb limit for massive images
Image.MAX_IMAGE_PIXELS = None


def chip_massive_image(image_path, output_dir, crop_size=224):
    """Crops a single large image into smaller tiles."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB to ensure consistency (removes Alpha channel if present)
            img = img.convert("RGB")
            width, height = img.size
            filename_stem = Path(image_path).stem  # Gets 'image_name' without extension

            os.makedirs(output_dir, exist_ok=True)

            # Grid cropping logic
            for x in range(0, width, crop_size):
                for y in range(0, height, crop_size):
                    box = (x, y, x + crop_size, y + crop_size)
                    chip = img.crop(box)

                    # Only save if the chip is the full size (skips edges)
                    if chip.size == (crop_size, crop_size):
                        chip_name = f"{filename_stem}_x{x}_y{y}.jpg"
                        chip.save(os.path.join(output_dir, chip_name), quality=95)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def process_gaia_dataset(input_root, output_root, crop_size=224):
    """Walks through GAIA structure and chips all images found in 'images' folders."""

    print(f"Starting chipping process. Looking in: {input_root}")

    for root, dirs, files in os.walk(input_root):
        # We check if the current folder is named 'images' (per your screenshot)
        if os.path.basename(root) == "images":

            # 2. Calculate the relative path to maintain structure
            # e.g., 'Atmosphere/air_pollution/images'
            rel_path = os.path.relpath(root, input_root)

            # 3. Define where the chips should go
            # e.g., 'data/GAIA_chipped/Atmosphere/air_pollution/images'
            target_output_dir = os.path.join(output_root, rel_path)

            print(f"Processing: {rel_path}")

            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    full_input_path = os.path.join(root, file)
                    chip_massive_image(full_input_path, target_output_dir, crop_size)


# --- EXECUTION ---
INPUT_DIR = "data/GAIA"  # Where your current data lives
OUTPUT_DIR = "data/GAIA_tiles"  # Where you want the chips to go

process_gaia_dataset(INPUT_DIR, OUTPUT_DIR, crop_size=224)
print("Done! You can now point your DataModule to 'data/GAIA_tiles'")