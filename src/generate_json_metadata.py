import os
import json
from glob import glob


def generate_tile_metadata(original_metadata_path, tiles_subclass_dir):
    with open(original_metadata_path, 'r') as f:
        original_data = json.load(f)

    # NEW: Point to the actual image subfolder
    actual_image_dir = os.path.join(tiles_subclass_dir, "images")

    if not os.path.exists(actual_image_dir):
        print(f"Skipping: No 'images' folder found in {tiles_subclass_dir}")
        return

    new_metadata = []

    for entry in original_data:
        parent_filename = os.path.splitext(os.path.basename(entry['image_path']))[0]

        # Search inside the 'images' subfolder
        search_pattern = os.path.join(actual_image_dir, f"{parent_filename}_x*.jpg")
        chips = glob(search_pattern)

        for chip_path in chips:
            # We want the path relative to where metadata.json will sit
            # So "images/chip_x0_y0.jpg"
            chip_rel_path = os.path.join("images", os.path.basename(chip_path))

            tile_entry = entry.copy()
            tile_entry['image_path'] = chip_rel_path
            tile_entry['parent_id'] = parent_filename
            new_metadata.append(tile_entry)

    # Save metadata.json in the subclass folder (Atmosphere/air_pollution/metadata.json)
    with open(os.path.join(tiles_subclass_dir, "metadata.json"), 'w') as f:
        json.dump(new_metadata, f, indent=4)

    print(f"Generated {len(new_metadata)} tile entries for {tiles_subclass_dir}")


import os

# --- SET YOUR PATHS HERE ---
ORIGINAL_GAIA_DIR = "/home/antonio/projects/mamba-gaia-ir/data/GAIA"
TILED_GAIA_DIR = "/home/antonio/projects/mamba-gaia-ir/data/GAIA_tiles"


def run_metadata_regeneration():
    for sphere in sorted(os.listdir(ORIGINAL_GAIA_DIR)):
        sphere_path = os.path.join(ORIGINAL_GAIA_DIR, sphere)
        if not os.path.isdir(sphere_path): continue

        for sub_class in sorted(os.listdir(sphere_path)):
            original_sub_path = os.path.join(sphere_path, sub_class)
            original_meta = os.path.join(original_sub_path, "metadata.json")

            # Check if original metadata exists
            if os.path.isfile(original_meta):
                # Define where the tiled subclass folder is
                tiled_sub_path = os.path.join(TILED_GAIA_DIR, sphere, sub_class)

                if os.path.exists(tiled_sub_path):
                    print(f"Processing: {sphere}/{sub_class}...")
                    generate_tile_metadata(original_meta, tiled_sub_path)
                else:
                    print(f"Skipping: {tiled_sub_path} does not exist.")


if __name__ == "__main__":
    run_metadata_regeneration()