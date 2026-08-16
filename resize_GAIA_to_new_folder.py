import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def resize_dataset(source_dir, target_dir, size=(512, 512)):
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 1. Get all JPG files
    image_files = list(source_path.rglob("*.jpg"))
    print(f"Found {len(image_files)} images. Starting resize to {target_dir}...")

    # 2. Process images
    for img_path in tqdm(image_files):
        # Create the relative path for the new folder
        rel_path = img_path.relative_to(source_path)
        new_path = target_path / rel_path

        # Create subdirectories if they don't exist
        new_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(img_path) as img:
                # Use Resampling.BOX for maximum speed on high-res downsizes
                img_resized = img.resize(size, resample=Image.Resampling.BOX)

                # Convert to RGB to ensure JPEG compatibility (removes Alpha if exists)
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')

                img_resized.save(new_path, "JPEG", quality=85)
        except Exception as e:
            print(f"Skipping corrupt image {img_path}: {e}")


if __name__ == "__main__":
    # Update these paths to match your system
    SOURCE = "/home/antonio/projects/mamba-gaia-ir/data/GAIA"
    TARGET = "/home/antonio/projects/mamba-gaia-ir/data/GAIA_512"

    resize_dataset(SOURCE, TARGET)
    print("\nDone! Now update your config to point to GAIA_512.")