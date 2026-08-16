import json
import requests
import os
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
import numpy as np
from itertools import combinations
from collections import Counter


def download_dataset_by_spheres(df, spheres_dict, output_dir='GAIA/'):
    """
    Downloads images organized by Earth Sphere and Tag.
    """
    for sphere, tags in spheres_dict.items():
        print(f"\n=== Processing Sphere: {sphere} ===")

        for tag in tags:
            # 1. Filter for the tag
            mask = df['tag'].apply(lambda x: tag in x if isinstance(x, list) else False)
            filtered_df = df[mask].copy()

            if filtered_df.empty:
                continue

            # 2. Setup Hierarchical Directories
            # Path: GAIA/val/Biosphere/wildfire/images/
            tag_dir = os.path.join(output_dir, sphere, tag)
            image_dir = os.path.join(tag_dir, 'images')
            os.makedirs(image_dir, exist_ok=True)

            new_json_structure = []
            print(f"Downloading images for [{tag}]...")

            # 3. Download Loop
            for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), leave=False):
                image_url = row['image_src']
                image_filename = f"{row['id']}.jpg"
                image_path_full = os.path.join(image_dir, image_filename)

                # Path for metadata relative to the tag folder
                json_path = f"images/{image_filename}"

                try:
                    if not os.path.exists(image_path_full):
                        response = requests.get(image_url, stream=True, timeout=10)
                        response.raise_for_status()
                        with open(image_path_full, 'wb') as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)

                    new_json_structure.append({
                        "image_id": row['id'],
                        "image_path": json_path,
                        "captions": row['captions'],
                        "all_tags": row['tag'],
                        "earth_sphere": sphere
                    })
                except Exception:
                    continue  # Skip failed downloads silently to keep logs clean

            # 4. Save Metadata for this specific tag
            if new_json_structure:
                output_json_path = os.path.join(tag_dir, 'metadata.json')
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(new_json_structure, f, indent=4)

    print("\n✅ Hierarchical download complete!")


def download_dataset_by_spheres_v2(df, output_dir):
    """
    Downloads images organized by the newly classified Earth Sphere.
    Structure: GAIA/[Sphere]/[Primary_Tag]/images/
    """
    # 1. Group by the new classification results
    # Ensure you use the correct column name here (e.g., 'earth_sphere')
    spheres = df['earth_sphere'].unique()

    for sphere in spheres:
        print(f"\n=== Processing Sphere: {sphere} ===")
        sphere_df = df[df['earth_sphere'] == sphere]

        # 2. Within each sphere, we organize by the image's first tag
        # to maintain a granular folder structure
        # (Using .apply to get the first tag as a string)
        sphere_df = sphere_df.copy()
        sphere_df['primary_tag'] = sphere_df['tag'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'misc')

        unique_tags = sphere_df['primary_tag'].unique()

        for tag in unique_tags:
            tag_filtered_df = sphere_df[sphere_df['primary_tag'] == tag]

            # 3. Setup Hierarchical Directories
            tag_dir = os.path.join(output_dir, str(sphere), str(tag).replace(" ", "_"))
            image_dir = os.path.join(tag_dir, 'images')
            os.makedirs(image_dir, exist_ok=True)

            new_json_structure = []
            print(f"Downloading {len(tag_filtered_df)} images for [{tag}]...")

            # 4. Download Loop
            for _, row in tqdm(tag_filtered_df.iterrows(), total=len(tag_filtered_df), leave=False):
                image_url = row['image_src']
                image_filename = f"{row['id']}.jpg"
                image_path_full = os.path.join(image_dir, image_filename)

                # Path for metadata relative to the tag folder
                json_rel_path = f"images/{image_filename}"

                try:
                    if not os.path.exists(image_path_full):
                        response = requests.get(image_url, stream=True, timeout=10)
                        response.raise_for_status()
                        with open(image_path_full, 'wb') as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)

                    new_json_structure.append({
                        "image_id": row['id'],
                        "image_path": json_rel_path,
                        "captions": row.get('captions', []),
                        "all_tags": row['tag'],
                        "classified_sphere": sphere
                    })
                except Exception:
                    continue

                    # 5. Save Metadata for this specific tag
            if new_json_structure:
                output_json_path = os.path.join(tag_dir, 'metadata.json')
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(new_json_structure, f, indent=4)

    print("\n✅ Hierarchical download complete!")


def download_dataset_by_spheres_v3(df, output_dir, min_count=100):
    """
    Downloads images organized by the newly classified Earth Sphere,
    filtering for tags that have more than 100 images.
    """
    # Define your areas of interest
    priority_spheres = ['Biosphere', 'Hydrosphere', 'Atmosphere']

    spheres = df['earth_sphere'].unique()

    for sphere in spheres:
        # --- THE EXCLUSION TWEAK ---
        if sphere not in priority_spheres:
            print(f"\n--- Skipping Sphere: {sphere} (Not a priority) ---")
            continue

        print(f"\n=== Processing Priority Sphere: {sphere} ===")
        sphere_df = df[df['earth_sphere'] == sphere].copy()

        # Grouping by the first tag in the list
        sphere_df['primary_tag'] = sphere_df['tag'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'misc')

        unique_tags = sphere_df['primary_tag'].unique()

        for tag in unique_tags:
            tag_filtered_df = sphere_df[sphere_df['primary_tag'] == tag]

            # --- THE ADDITION: Threshold Check ---
            tag_count = len(tag_filtered_df)
            if tag_count <= min_count:
                # Optional: print(f"Skipping [{tag}]: Only {tag_count} images.")
                continue

            # 3. Setup Hierarchical Directories
            tag_dir = os.path.join(output_dir, str(sphere), str(tag).replace(" ", "_"))
            image_dir = os.path.join(tag_dir, 'images')
            os.makedirs(image_dir, exist_ok=True)

            new_json_structure = []
            print(f"Downloading {tag_count} images for [{tag}]...")

            # 4. Download Loop
            for _, row in tqdm(tag_filtered_df.iterrows(), total=tag_count, leave=False):
                image_url = row['image_src']
                image_filename = f"{row['id']}.jpg"
                image_path_full = os.path.join(image_dir, image_filename)
                json_rel_path = f"images/{image_filename}"

                try:
                    if not os.path.exists(image_path_full):
                        response = requests.get(image_url, stream=True, timeout=10)
                        response.raise_for_status()
                        with open(image_path_full, 'wb') as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)

                    new_json_structure.append({
                        "image_id": row['id'],
                        "image_path": json_rel_path,
                        "captions": row.get('captions', []),
                        "all_tags": row['tag'],
                        "classified_sphere": sphere
                    })
                except Exception:
                    continue

            # 5. Save Metadata (Inside tag loop)
            if new_json_structure:
                output_json_path = os.path.join(tag_dir, 'metadata.json')
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(new_json_structure, f, indent=4)

    print("\n✅ Hierarchical download complete (Filtered for >100 images)!")

def download_by_tag(df, target_tag='wildfire', output_dir='GAIA/val',
                    image_folder='single_tag_check_images', output_json_name='output.json'):
    """
    Uses the cleaned DataFrame to filter by tag and download images.
    """

    # --- 1. Filter the DataFrame ---
    # Since we are passing 'df' directly, we just use your existing cleaned 'tag' column
    mask = df['tag'].apply(lambda x: target_tag in x if isinstance(x, list) else False)
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        print(f"No images found containing the tag: '{target_tag}'")
        return

    # # Limit to 100 images as per your previous requirement
    num_elements = max(len(filtered_df), 100)
    filtered_df = filtered_df.head(num_elements)

    # --- 2. Setup Directories ---
    tag_dir = os.path.join(output_dir, target_tag)
    image_dir = os.path.join(tag_dir, image_folder)
    os.makedirs(image_dir, exist_ok=True)

    # --- 3. Download Loop ---
    new_json_structure = []
    print(f"Downloading {num_elements} images for '{target_tag}'...")

    for _, row in tqdm(filtered_df.iterrows(), total=num_elements):
        image_url = row['image_src']
        image_filename = f"{row['id']}.jpg"
        image_path_full = os.path.join(image_dir, image_filename)

        # Relative path for the output JSON
        json_path = os.path.join(target_tag, image_folder, image_filename).replace(os.path.sep, '/')

        try:
            if not os.path.exists(image_path_full):
                response = requests.get(image_url, stream=True, timeout=10)
                response.raise_for_status()
                with open(image_path_full, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)

            new_json_structure.append({
                "image_path": json_path,
                "captions": row['captions'],
                "classes": row['tag']
            })

        except Exception as e:
            print(f"Failed to download image {row['id']}: {e}")

    # --- 4. Save Metadata ---
    output_json_path = os.path.join(tag_dir, output_json_name)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_json_structure, f, indent=4)

    print(f"\nSuccess! Images saved to: {image_dir}")


def download_images_and_create_json(input_file_path='input.json', output_dir='GAIA/val', image_folder='images',
                                    output_json_name='output.json'):
    """
    Downloads images from a JSON file, saves them, and creates a new JSON
    structure with image paths and captions.

    Args:
        input_file_path (str): Path to the original JSON file.
        output_dir (str): The main directory to store data (e.g., 'data').
        image_folder (str): The sub-folder inside output_dir for images (e.g., 'images').
        output_json_name (str): The name of the new JSON file.
    """

    # --- 1. Load the original JSON data ---
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file_path}'.")
        return

    # Assuming the fields are lists of the same length
    image_sources = data.get('image_src', [])
    captions_list = data.get('captions', [])
    image_ids = data.get('id', [])
    classes = data.get('tag', [])

    if not image_sources:
        print("Error: 'image_src' field is empty or missing in the JSON.")
        return

    if len(image_sources) != len(captions_list):
        print("Warning: 'image_src' and 'captions' fields have different lengths. Using the shorter length.")

    # Determine the number of elements to process
    # num_elements = min(len(image_sources), len(captions_list))
    num_elements = 100

    # --- 2. Setup Directories ---
    # Create 'data/images' folder structure
    image_dir = os.path.join(output_dir, image_folder)
    os.makedirs(image_dir, exist_ok=True)
    print(f"Directories created/verified: '{image_dir}'")

    # --- 3. Download Images and Prepare New JSON Structure ---
    new_json_structure = []

    print(f"Starting download of {num_elements} images...")

    for i in tqdm(range(num_elements)):
        image_url = image_sources[i]

        # Construct the image filename (e.g., '0.jpg', '1.jpg', etc.)
        # A simple way to get the file extension is to check the URL,
        # but for simplicity and consistency, we'll use a default or
        # assume a common extension like '.jpg'.
        # A more robust solution might parse the actual extension from the URL.
        # Here we'll just use a fixed '.jpg' for numbering consistency.
        image_filename = f"{image_ids[i]}.jpg"
        image_path_full = os.path.join(image_dir, image_filename)

        # This is the path we want in the new JSON (relative to the script's execution context)
        json_path = os.path.join(output_dir, image_folder, image_filename).replace(os.path.sep, '/')

        try:
            # Download the image content
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            # Save the image content
            with open(image_path_full, 'wb') as image_file:
                for chunk in response.iter_content(1024):
                    image_file.write(chunk)

            # Append to the new JSON structure
            new_json_structure.append({
                "image_path": json_path,
                "captions": captions_list[i],
                "classes": classes
            })

            if (i + 1) % 100 == 0:
                print(f"Downloaded {i + 1}/{num_elements} images.")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading image {i} from {image_url}: {e}")
            # Optionally, you could skip this image or log the error and continue
        except Exception as e:
            print(f"An unexpected error occurred while processing image {i}: {e}")

    print("\n--- Download complete ---")

    # --- 4. Save the new JSON structure ---
    output_json_path = os.path.join(output_dir, output_json_name)
    try:
        with open(output_json_path, 'w') as f:
            json.dump(new_json_structure, f, indent=4)
        print(f"Successfully created new JSON file at: **{output_json_path}**")
        print(f"The 'images' folder is located at: **{image_dir}**")
    except Exception as e:
        print(f"Error saving the new JSON file: {e}")


def download_agri_dataset_by_tag(df, output_dir):
    """
    Creates a folder for each agriculture-related tag and downloads relevant images.
    """
    # These are the folder names that will appear inside GAIA_Agri
    target_agri_tags = [
        'agriculture',
        'irrigation_systems',
        'paddy_fields',
        'vineyards_and_orchards',
        'plantations',
        'greenhouses',
        'agricultural_cycle',
        'pasture_and_rangeland',
        'agricultural_expansion',
        'agricultural_burning'
    ]

    # Keywords to ensure we don't miss images with agri-content in captions
    agri_keywords = ['farm', 'crop', 'harvest', 'plow', 'orchard', 'vineyard']

    print(f"Building Agricultural Dataset in: {output_dir}")

    for folder_name in target_agri_tags:
        # 1. Filter the cleaned 'tag' column for this specific folder_name
        mask_tags = df['tag'].apply(lambda x: folder_name in x if isinstance(x, list) else False)

        filtered_df = df[mask_tags].copy()

        if filtered_df.empty:
            print(f"Skipping {folder_name}: No images found.")
            continue

        # 2. Create the specific subfolder: D:/<output_dir>/<folder_name>/images/
        tag_dir = os.path.join(output_dir, folder_name)
        image_dir = os.path.join(tag_dir, 'images')
        os.makedirs(image_dir, exist_ok=True)

        new_json_structure = []
        print(f"Processing folder [{folder_name}] - {len(filtered_df)} images found.")

        # 3. Download Loop
        for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), leave=False):
            image_filename = f"{row['id']}.jpg"
            image_path_full = os.path.join(image_dir, image_filename)

            try:
                if not os.path.exists(image_path_full):
                    response = requests.get(row['image_src'], stream=True, timeout=15)
                    response.raise_for_status()
                    with open(image_path_full, 'wb') as f:
                        for chunk in response.iter_content(8192):
                            f.write(chunk)

                new_json_structure.append({
                    "image_id": row['id'],
                    "image_path": f"images/{image_filename}",
                    "captions": row['captions'],
                    "tags": row['tag']  # Includes all cleaned tags for the model to see
                })
            except Exception:
                continue

                # 5. Save metadata.json inside D:/GAIA_Agri/<folder_name>/
        if new_json_structure:
            meta_path = os.path.join(tag_dir, 'metadata.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(new_json_structure, f, indent=4)

    print(f"\n✅ All agricultural folders created at {output_dir}")


def download_agri_master_dataset(df_universe, output_dir=r'D:/GAIA_Agri_Final'):
    """
    Downloads unique images into a single folder and creates a master metadata.json.
    """
    image_dir = os.path.join(output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    master_metadata = []
    metadata_path = os.path.join(output_dir, 'metadata.json')

    print(f"🚀 Downloading {len(df_universe)} unique agricultural images...")

    for _, row in tqdm(df_universe.iterrows(), total=len(df_universe)):
        image_id = row['id']
        image_filename = f"{image_id}.jpg"
        save_path = os.path.join(image_dir, image_filename)

        # 1. Download Logic (Unique files only)
        if not os.path.exists(save_path):
            try:
                response = requests.get(row['image_src'], timeout=15)
                response.raise_for_status()
                with open(save_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Skipping {image_id} due to error: {e}")
                continue

        # 2. Metadata Logic (Flatten sets to lists for JSON compatibility)
        master_metadata.append({
            "image_id": image_id,
            "file_path": f"images/{image_filename}",
            "subspheres": list(row['found_subspheres']),
            "original_tags": row['tag'],
            "caption": row['captions']
        })

    # 3. Save Master JSON
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(master_metadata, f, indent=4)

    print(f"✅ Success! Data saved to {output_dir}")
    print(f"Total Images: {len(df_universe)}")
    print(f"Master Metadata: {metadata_path}")

# This was made up after EDA on the top 126 tags (the ones that have a frequency > 100)
cleaning_map = {
    'wildfires': 'wildfire',
    'bushfires': 'wildfire',
    'volcanoes': 'volcano',
    'volcanic activity': 'volcano',
    'volcanic eruption': 'volcano',
    'eruption': 'volcano',
    'smoke plumes': 'smoke plume',
    'smoke': 'smoke plume',
    'dust storms': 'dust storm',
    'flooding': 'floods',
    'urban area': 'urban',
    'cyclone': 'tropical_cyclone',
    'typhoon': 'tropical_cyclone',
    'hurricane': 'tropical_cyclone',
    'icebergs': 'iceberg',
    'glaciers': 'glacier',
    'atmospheric phenomenon': 'atmospheric phenomena',
    'terra satellite': 'terra',
    'aqua satellite': 'aqua'
}

semantic_merge_map = {
    'fires': 'wildfire',
    'tropical cyclone': 'tropical_cyclone',
    'tropical storm': 'tropical_cyclone',
    'dust and haze': 'dust storm'  # Optional: if you want to group all atmospheric dust
}

master_clean_map = {
    # --- ATMOSPHERE & POLLUTION ---
    'atmospheric phenomena': 'atmosphere',
    'atmospheric phenomenon': 'atmosphere',
    'atmospheric conditions': 'atmosphere',
    'air quality': 'air_pollution',
    'air pollution': 'air_pollution',
    'pollution': 'air_pollution',
    'aerosols': 'air_pollution',
    'dust storm': 'dust_storm',
    'dust storms': 'dust_storm',
    'dust and haze': 'dust_storm',
    'dust': 'dust_storm',
    'haze': 'dust_storm',
    'saharan dust': 'dust_storm',
    'sahara desert': 'dust_storm',
    'smoke plumes': 'smoke_plume',
    'smoke': 'smoke_plume',
    'smoke plume': 'smoke_plume',
    'plume': 'smoke_plume',

    # --- FIRE & VOLCANOES ---
    'wildfires': 'wildfire',
    'fires': 'wildfire',
    'agricultural fires': 'wildfire',
    'burn scars': 'wildfire',
    'bushfires': 'wildfire',
    'volcano': 'volcanic_activity',
    'volcanoes': 'volcanic_activity',
    'volcanic activity': 'volcanic_activity',
    'volcanic eruption': 'volcanic_activity',
    'volcanic plume': 'volcanic_activity',
    'eruption': 'volcanic_activity',
    'ash plume': 'volcanic_activity',

    # --- CRYOSPHERE (ICE/SNOW) ---
    'snow cover': 'snow',
    'snowfall': 'snow',
    'snowice': 'snow',
    'ice shelf': 'iceberg',
    'icebergs': 'iceberg',
    'glaciers': 'glacier',

    # --- WATER & COASTAL ---
    'flooding': 'floods',
    'hydrology': 'water',
    'lakes': 'water',
    'rivers': 'water',
    'great lakes': 'water',
    'coastline': 'coastal',
    'phytoplankton bloom': 'phytoplankton',
    'algal bloom': 'phytoplankton',
    'sediment transport': 'sediment',

    # --- DISASTERS & STORMS ---
    'natural disaster': 'natural_disaster',
    'natural disasters': 'natural_disaster',
    'natural events': 'natural_disaster',
    'weather event': 'natural_disaster',
    'severe weather': 'natural_disaster',
    'severe storms': 'natural_disaster',
    'tropical cyclone': 'tropical_cyclone',
    'cyclone': 'tropical_cyclone',
    'tropical storm': 'tropical_cyclone',
    'typhoon': 'tropical_cyclone',
    'hurricane': 'tropical_cyclone',

    # --- LAND & HUMAN ---
    'land use': 'land_use',
    'land cover': 'land_use',
    'land monitoring': 'land_use',
    'landscape': 'land_use',
    'urban area': 'urban',

    # --- SATELLITE MISSIONS ---
    'terra satellite': 'terra',
    'aqua satellite': 'aqua'
}

scientific_map = {
    # --- The Atmosphere / Weather Core ---
    'meteorology': 'atmosphere',
    'weather': 'atmosphere',
    'atmospheric phenomena': 'atmosphere',
    'cloud patterns': 'atmosphere',
    'cloud streets': 'atmosphere',
    'atmospheric conditions': 'atmosphere',
    'rainfall': 'precipitation',
    'monsoon': 'precipitation',

    # --- The Climate & Environment Core ---
    'climate change': 'climate',
    'environmental impact': 'environment_change',
    'environmental monitoring': 'environment_change',
    'biodiversity': 'environment_change',

    # --- The Water & Coastal Core ---
    'coastline': 'coastal',
    'marine ecosystem': 'oceanography',
    'red sea': 'oceanography',
    'arabian sea': 'oceanography',
    'indian ocean': 'oceanography',
    'pacific ocean': 'oceanography',
    'atlantic ocean': 'oceanography',
    'sediment': 'hydrology',
    'wetlands': 'hydrology',
    'water': 'hydrology',

    # --- The Cryosphere Core (Optional: Merge only if you want a broad view) ---
    'snow cover': 'snow',
    'sea ice': 'cryosphere',
    'iceberg': 'cryosphere',
    'glacier': 'cryosphere',
    'arctic': 'polar_regions',
    'antarctica': 'polar_regions',
    'greenland': 'polar_regions',

    # --- The Land & Human Core ---
    'land use': 'land_management',
    'land_use': 'land_management',
    'agriculture': 'land_management',
    'infrastructure': 'urban',
    'human': 'urban',
    'landscape': 'land'
}

# earth_spheres = {
#     'Atmosphere': [
#         'atmosphere', 'tropical_cyclone', 'dust_storm', 'air_pollution',
#         'precipitation', 'smoke_plume', 'climate'
#     ],
#     'Hydrosphere': [
#         'water', 'floods', 'hydrology', 'oceanography', 'phytoplankton',
#         'coastal', 'sediment'
#     ],
#     'Biosphere': [
#         'vegetation', 'deforestation', 'wildfire', 'agriculture',
#         'land_use', 'drought'
#     ],
#     'Geosphere': [
#         'geology', 'topography', 'desert', 'volcanic_activity',
#         'land_management', 'urban', 'natural_disaster'
#     ],
#     'Cryosphere': [
#         'cryosphere', 'polar_regions', 'glacier', 'iceberg',
#         'snow', 'winter', 'alaska', 'siberia'
#     ]
# }


earth_spheres = {
    'Atmosphere': [
        '3D Cloud Localization', '3D Rainfall', 'Adiabatic cooling', 'Aerosol', 'Aerosol Cloud',
        'Aerosol Dispersion', 'Aerosol Effects', 'Aerosol Emissions', 'Aerosol Haze', 'Aerosol Impact',
        'Aerosol Index', 'Aerosol Monitoring', 'Aerosol Movement', 'Aerosol Particles', 'Aerosol Plume',
        'Aerosol Plumes', 'Aerosol Pollution', 'Aerosol Science', 'Aerosol Transport', 'Aerosol clouds',
        'Aerosol effects', 'Aerosol impact', 'Aerosol interactions', 'Aerosol particles', 'Aerosol transport',
        'Aerosol-cloud interactions', 'Aerosols', 'African Dust', 'African Dust Storms', 'Air Pollution',
        'Air Quality', 'Airborne Dust', 'Aircraft Dissipation Trails', 'Airglow', 'Albedo',
        'Alberta Clipper', 'Ammonia', 'Anvil Cloud', 'Anvil Clouds', 'Arctic Air', 'Arctic Blast',
        'Arctic Haze', 'Arctic Oscillation', 'Arctic air mass', 'Arctic chill', 'Arctic stratosphere',
        'Ash Cloud', 'Ash Dispersal', 'Ash Dispersion', 'Ash Emission', 'Ash Emissions', 'Ash Fallout',
        'Ash Plume', 'Ash Plumes', 'Ash Vortices', 'Asian Brown Cloud', 'Asian Dust', 'Atmosphere',
        'Atmospheric', 'Atmospheric Aerosols', 'Atmospheric Analysis', 'Atmospheric Circulation',
        'Atmospheric Composition', 'Atmospheric Conditions', 'Atmospheric Disturbance', 'Atmospheric Dust',
        'Atmospheric Dynamics', 'Atmospheric Effects', 'Atmospheric Event', 'Atmospheric Events',
        'Atmospheric Gravity Waves', 'Atmospheric Haze', 'Atmospheric Inversion', 'Atmospheric Monitoring',
        'Atmospheric Motion', 'Atmospheric Observation', 'Atmospheric Particles', 'Atmospheric Particulate',
        'Atmospheric Patterns', 'Atmospheric Phenomena', 'Atmospheric Phenomenon', 'Atmospheric Pollution',
        'Atmospheric Pressure', 'Atmospheric Processes', 'Atmospheric Radiation', 'Atmospheric River',
        'Atmospheric Rivers', 'Atmospheric Science', 'Atmospheric Smoke', 'Atmospheric Transport',
        'Atmospheric Turbulence', 'Atmospheric Waves', 'Atmospheric chemistry', 'Atmospheric convection',
        'Aurora', 'Aurora Australis', 'Aurora Borealis', 'Auroras', 'Blizzard', 'Bomb Cyclone',
        'Carbon Dioxide', 'Carbon Monoxide', 'Cirrus Clouds', 'Climate', 'Cloud', 'Cloud Analysis',
        'Cloud Cover', 'Cloud Dynamics', 'Cloud Formation', 'Cloud Patterns', 'Cloud Streets',
        'Cloud Vortex', 'Cold Front', 'Cyclone', 'Dust Storm', 'Extreme Weather', 'Hurricane',
        'Jet Stream', 'Lightning', 'Meteorology', 'Monsoon', 'Ozone', 'Precipitation', 'Smog',
        'Storm', 'Temperature', 'Weather', 'Wind'
    ],
    'Biosphere': [
        'Active Fire', 'Active Fires', 'Agricultural', 'Agricultural Areas', 'Agricultural Burning',
        'Agricultural Damage', 'Agricultural Development', 'Agricultural Expansion', 'Agricultural Fires',
        'Agricultural Impact', 'Agricultural Land', 'Agricultural Land Use', 'Agricultural Monitoring',
        'Agricultural Plains', 'Agricultural Practices', 'Agricultural Region', 'Agricultural Run-off',
        'Agricultural Runoff', 'Agricultural Zones', 'Agricultural burning', 'Agricultural fires',
        'Agricultural monitoring', 'Agricultural patterns', 'Agricultural practices', 'Agriculture',
        'Agriculture Impact', 'Agriculture fires', 'Algae', 'Algae Bloom', 'Algae Blooms', 'Algal Bloom',
        'Algal Blooms', 'Amazon Rainforest', 'Anaerobic Bacteria', 'Animal Tracking', 'Antarctic Wildlife',
        'Aquaculture', 'Aquatic Ecosystem', 'Aquatic Plants', 'Aquatic Vegetation', 'Arctic Ecosystem',
        'Arctic Vegetation', 'Atlantic Forest', 'Atlantic Rainforest', 'Autumn Colors', 'Autumn Foliage',
        'Avian Influenza', 'Banana Plantation', 'Bialowieza National Park', 'Biodiversity', 'Biological',
        'Biomass', 'Biomass Burning', 'Biomes', 'Biosphere', 'Biosphere Reserve', 'Bird Habitat',
        'Boreal Forest', 'Brown Bears', 'Cactus Conservation', 'Carbon Absorption', 'Carbon Biomass',
        'Carbon Sequestration', 'Carbon Sink', 'Caribou', 'Cash Crops', 'Cattle Ranching', 'Cerrado',
        'Chaparral', 'Chlorophyll', 'Chlorophyll Bloom', 'Chlorophyll Concentration', 'Cicada Emergence',
        'Citrus Growing', 'Coconut Plantation', 'Coffee Plantation', 'Conifer Forests', 'Crop Analysis',
        'Crop Burning', 'Crop Damage', 'Crop Fires', 'Crop Mapping', 'Croplands', 'Crops', 'Cyanobacteria',
        'Deciduous Forests', 'Defoliation', 'Deforestation', 'Dengue fever', 'Diatoms', 'Dinoflagellates',
        'Ecological Diversity', 'Ecology', 'Ecosystem', 'Elephants', 'Endemic Species', 'Everglades',
        'Farming', 'Fish Farms', 'Fisheries', 'Flamingos', 'Flower Fields', 'Forest', 'Forest Fire',
        'Forestry', 'Fynbos', 'Grassland', 'Mangroves', 'Plankton', 'Vegetation', 'Wildlife'
    ],
    'Cryosphere': [
        'Adelie Coast', 'Amery Ice Shelf', 'Amundsen Gulf', 'Amundsen Sea', 'Antarctic', 'Antarctic Ice',
        'Antarctic Ice Flow', 'Antarctic Ice Shelf', 'Antarctic Sea Ice', 'Antarctic warming',
        'Antarctica', 'Arctic', 'Arctic Change', 'Arctic Cold', 'Arctic Conditions', 'Arctic Expedition',
        'Arctic Ocean', 'Arctic Region', 'Arctic Sea Ice', 'Arctic Thaw', 'Arctic Thawing', 'Arctic ice',
        'Arctic sea ice', 'Autumn Ice', 'Autumn Snow', 'Baffin Bay', 'Barents Sea', 'Beaufort Sea',
        'Bellingshausen Sea', 'Bering Sea', 'Bering Strait', 'Blowing Snow', 'Brunt Ice Shelf',
        'Calving', 'Calving Event', 'Calving Front', 'Canadian Arctic', 'Coastal Ice', 'Columbia Glacier',
        'CryoSat', 'Cryosphere', 'Debris-Covered Glaciers', 'Deglaciation', 'Early Snowfall',
        'Early Winter', 'East Antarctica', 'Epishelf Lake', 'Fast Ice', 'Fedchenko', 'Frozen',
        'Frozen Bay', 'Frozen Lake', 'Frozen Lakes', 'Frozen Rivers', 'Frozen Waters', 'Glacial',
        'Glacial Dynamics', 'Glacial Erosion', 'Glacial Flood', 'Glacial Lake', 'Glacial Melt',
        'Glacial Movement', 'Glacial Retreat', 'Glacial Runoff', 'Glaciation', 'Glacier',
        'Glacier Bay', 'Glacier Calving', 'Glacier Collapse', 'Glacier Flow', 'Glacier Melt',
        'Glacier National Park', 'Glaciers', 'Glaciology', 'Ice', 'Ice Shelf', 'Iceberg',
        'Permafrost', 'Ross Ice Shelf', 'Sea Ice', 'Snow', 'Snow Cover', 'Tundra'
    ],
    'Geosphere': [
        'Adrar Plateau', 'Aeolian Processes', 'Afar Depression', 'Ahaggar Mountains', 'Alaid Volcano',
        'Al Hajar Mountains', 'Alaska Range', 'Alluvial Fans', 'Alps', 'Altai Mountains', 'Altiplano',
        'Ambrym Volcano', 'Anak Krakatau', 'Anatahan Volcano', 'Andean Volcanic Belt', 'Andes Mountains',
        'Apennine Range', 'Appalachian Mountains', 'Arid Land', 'Arid Terrain', 'Artisanal Mining',
        'Aseismic Slip', 'Astrobleme', 'Atacama Desert', 'Atlas Mountains', 'Basalt Plateau',
        'Basaltic Lava Flows', 'Basin', 'Barchan Dunes', 'Badlands', 'Bighorn Mountains', 'Black Rock Desert',
        'Blue Ridge Mountains', 'Brooks Range', 'Burn Scar', 'Caldera', 'Canyon', 'Carpathian Mountains',
        'Cascade Range', 'Caucasus Mountains', 'Cinder Cones', 'Cliffs', 'Coal Mining', 'Coastal Geomorphology',
        'Colorado Plateau', 'Continental Divide', 'Copper Mining', 'Crater', 'Deccan Plateau',
        'Dendritic Patterns', 'Desert', 'Diamond Mining', 'Digital Tectonic Activity Map',
        'Dinosaur Fossils', 'Dune Field', 'Dunes', 'Earthquake', 'Earthquake Analysis',
        'Earthquake faults', 'Earthquakes', 'Elevation', 'Emi Koussi', 'Erosion', 'Eruption',
        'Fault Line', 'Fault Lines', 'Fossils', 'Fuego Volcano', 'Geohazard', 'Geologic Features',
        'Geologic Mapping', 'Geological Activity', 'Geological Formation', 'Geology', 'Geomorphology',
        'Geothermal', 'Geysers', 'Gobi Desert', 'Gold Mining', 'Grand Canyon', 'Mountains',
        'Plateau', 'Rock', 'Sediment', 'Soil', 'Tectonic', 'Volcano'
    ],
    'Hydrosphere': [
        'Acidic Lake', 'Adriatic Sea', 'Aegean Sea', 'Agulhas Current', 'Agusan River', 'Alboran Sea',
        'Alkaline Lake', 'Alpine Lake', 'Amazon River', 'Amazon River Delta', 'Amu Darya River',
        'Andaman Sea', 'Angara River', 'Antarctic Circumpolar Current', 'Aquamarine Waters',
        'Aquifer', 'Aquifer Depletion', 'Aquifer System', 'Arabian Sea', 'Arafura Sea', 'Aral Sea',
        'Arctic Waters', 'Argentine Sea', 'Arkansas River', 'Artificial Lake', 'Atlantic Ocean',
        'Atoll', 'Ayeyarwady River', 'Bahama Banks', 'Baltic Sea', 'Banda Sea', 'Barrier Reef',
        'Bathymetry', 'Bay', 'Bay of Bengal', 'Bay of Biscay', 'Beach', 'Benguela Current',
        'Benguela Upwelling', 'Bering Sea', 'Betsiboka Estuary', 'Black Sea', 'Black Water',
        'Bo Hai', 'Bosphorus Strait', 'Brackish Water', 'Brahmaputra River', 'Braided River',
        'Bristol Bay', 'Buzzards Bay', 'Caribbean Sea', 'Caspian Sea', 'Celebes Sea', 'Celtic Sea',
        'Chesapeake Bay', 'Choked Lagoon', 'Chukchi Sea', 'Closed-basin lake', 'Coastal Currents',
        'Coastal Estuaries', 'Coastal Flooding', 'Coastal Lagoons', 'Coastal Oceanography',
        'Coastal Waters', 'Colorado River', 'Congo River', 'Coral Reef', 'Coral Sea', 'Currents',
        'Dam', 'Danube Delta', 'Danube River', 'Dead zone', 'Delta', 'Dnieper River', 'Drainage',
        'Drake Passage', 'Drought', 'East Australian Current', 'East China Sea', 'Ebro Delta',
        'Ebro River', 'Eddies', 'Elbe River', 'English Channel', 'Estuary', 'Euphrates River',
        'Eutrophication', 'Evaporation', 'Flooding', 'Freshwater', 'Ganges River', 'Hydrology',
        'Lake', 'Ocean', 'River', 'Water'
    ]
}

satellite_mapping = {
    # ISS Merging
    'International Space Station': 'ISS',
    'International Space Station (ISS)': 'ISS',
    'STS-97 Space Shuttle': 'Space Shuttle',
    'Space Shuttle Endeavour': 'Space Shuttle',

    # Terra/Aqua Merging
    'NASA Terra': 'Terra',
    'NASA\'s Terra': 'Terra',
    'EOS-Terra': 'Terra',
    'NASA Aqua': 'Aqua',
    'NASA\'s Aqua': 'Aqua',

    # Sentinel Merging
    'Copernicus Sentinel-2': 'Sentinel-2',
    'Sentinel-2A': 'Sentinel-2',
    'Sentinel-2B': 'Sentinel-2',
    'Copernicus Sentinel-1': 'Sentinel-1',
    'Sentinel-1A': 'Sentinel-1',
    'Sentinel-1B': 'Sentinel-1',

    # Landstat Merging
    'Landsat-8': 'Landsat 8',
    'Landsat-7': 'Landsat 7',
}

modality_mapping = {
    # --- RADAR & SAR ---
    "synthetic aperture radar (sar)": "sar",
    "synthetic aperture radar": "sar",
    "synthetic-aperture radar (sar)": "sar",
    "sar": "sar",
    "radar": "radar",
    "active radar": "radar",
    "radar interferometry": "insar",
    "interferometric synthetic aperture radar (insar)": "insar",
    "insar": "insar",

    # --- OPTICAL & MULTISPECTRAL ---
    "visible": "optical",
    "panchromatic": "optical",
    "digital camera": "optical",
    "photograph": "optical",
    "multispectral": "multispectral",
    "multi-spectral": "multispectral",
    "hyperspectral": "hyperspectral",
    "spectral": "multispectral",

    # --- INFRARED & THERMAL ---
    "thermal": "infrared",
    "thermal infrared": "infrared",
    "near-infrared": "nir",
    "near infrared": "nir",
    "shortwave infrared": "swir",
    "short-wave infrared": "swir",
    "swir": "swir",

    # --- PRECIPITATION & WEATHER ---
    "multisatellite precipitation analysis": "precipitation analysis",
    "multi-satellite precipitation analysis": "precipitation analysis",
    "precipitation radar": "precipitation analysis",
    "precipitation": "precipitation analysis",
    "water vapor": "atmospheric",

    # --- ELEVATION & POSITION ---
    "digital elevation model": "dem",
    "dem": "dem",
    "lidar": "lidar",
    "laser altimetry": "lidar",
    "altimetry": "altimetry",
    "topographic": "elevation",

    # --- GROUND TRUTH & MODELS ---
    "in situ measurement": "in-situ",
    "in situ": "in-situ",
    "in-situ": "in-situ",
    "model": "simulation/model",
    "numerical model": "simulation/model"
}

# Use this mapping ALONE for your specialized GAIA_Agri project
agri_specialized_map = {
    # --- BROAD AGRICULTURE ---
    'agriculture': 'agriculture',
    'agricultural': 'agriculture',
    'agricultural impact': 'agriculture',
    'land use': 'agriculture',
    'land cover': 'agriculture',
    'cultivated_land': 'agriculture',
    'croplands': 'agriculture',

    # --- WETLANDS & PADDYS ---
    'wetlands': 'wetlands_and_paddys',
    'coastal wetlands': 'wetlands_and_paddys',
    'paddy_fields': 'wetlands_and_paddys',

    # --- AGRI-FIRE (Important for Remote Sensing) ---
    'agricultural fires': 'agricultural_burning',
    'burn scars': 'agricultural_burning',

    # --- THE IMPACT ---
    'agricultural expansion': 'agricultural_expansion',
    'deforestation': 'agricultural_expansion'
}


def apply_mapping(tag_list, mapping):
    if not isinstance(tag_list, list): return []
    # Strip spaces, lowercase, map the word, and use set() to remove new duplicates
    cleaned = {mapping.get(t.strip().lower(), t.strip().lower()) for t in tag_list}
    return sorted(list(cleaned))


def check_tag_context(df, target, n=15):
    """Visualizes which tags appear most often alongside a target tag."""
    if target not in df.columns:
        print(f"Tag '{target}' not found in DataFrame.")
        return

    # Calculate co-occurrence and drop the target itself
    counts = df[df[target] == 1].sum().drop(target).sort_values(ascending=False).head(n)

    clean_labels = [str(label).replace('_', ' ').title() for label in counts.index]

    plt.figure(figsize=(13, 7))
    sns.barplot(x=counts.values, y=clean_labels, hue=counts.index, palette='magma', legend=False)
    plt.title(f"Top {n} Tags Co-occurring with '{target.replace('_', ' ').title()}'", fontsize=14)
    plt.xlabel("Total Image Count")
    plt.ylabel(None)  # Removes the "None" or "Index" label from the side
    plt.grid(axis='x', linestyle='--', alpha=0.6)


def classify_to_sphere(image_tags):
    tag_blob = " ".join([str(t).lower() for t in image_tags])

    anchors = {
        'Atmosphere': ['dust_storm', 'cyclon', 'typhoon', 'hurricane', 'tornado', 'smoke_plume', 'aerosol', 'cloud',
                       'ash', 'haze', 'meteorology', 'weather_pattern'],
        'Hydrosphere': ['water', 'ocean', 'sea', 'river', 'lake', 'flood', 'delta', 'estuar', 'phytoplankton', 'algae',
                        'marine', 'coast', 'reef', 'reservoir', 'currents', 'hydrology', 'tidal'],
        'Biosphere': ['agri', 'farm', 'crop', 'forest', 'veget', 'plant', 'fire', 'burn', 'wildfire', 'ndvi', 'paddy',
                      'tree', 'leaf', 'flora', 'harvest', 'deforest', 'ecology', 'habitat', 'irrigat', 'cultiv',
                      'plantation', 'orchard'],
        'Cryosphere': ['ice', 'snow', 'glacier', 'polar', 'arctic', 'freeze', 'frost', 'permafrost', 'antarct',
                       'iceberg', 'shelf', 'meltwater'],
        'Geosphere': ['geology', 'mining', 'volcan', 'earthq', 'soil', 'mountain', 'topography', 'urban', 'city',
                      'plateau', 'desert', 'land_use', 'land_management', 'geography']
    }

    scores = {sphere: 0 for sphere in anchors}

    for sphere, stems in anchors.items():
        for stem in stems:
            if stem in tag_blob:
                scores[sphere] += 2

                # --- STRATEGIC WEIGHTING ---
    if scores['Biosphere'] > 0: scores['Biosphere'] += 5  # Maximum protection for your top interest
    if scores['Hydrosphere'] > 0: scores['Hydrosphere'] += 2
    if scores['Atmosphere'] > 0: scores['Atmosphere'] += 1

    # Geosphere Tax: Only wins if it's the ONLY clear signal
    if scores['Geosphere'] > 0: scores['Geosphere'] -= 2

    top_sphere = max(scores, key=scores.get)

    if scores[top_sphere] <= 0:
        if any(c in tag_blob for c in ['cold', 'winter', 'degree']):
            return 'Cryosphere'
        return 'Geosphere'

    return top_sphere


# Count how many distinct spheres are represented in each image
def count_spheres(image_tags):
    found_spheres = set()
    for sphere, sphere_tags in earth_spheres.items():
        if set(image_tags) & set(sphere_tags):
            found_spheres.add(sphere)
    return len(found_spheres)


def get_all_spheres(image_tags):
    found = set()
    for sphere, sphere_tags in earth_spheres.items():
        if set(image_tags) & set(sphere_tags):
            found.add(sphere)
    return sorted(list(found))


def apply_mapping_robust(tag_list, mapping):
    if not isinstance(tag_list, list):
        return []

    # 1. Clean the tag (lower, strip)
    # 2. Map it if it exists in dict, else keep original
    # 3. Use set() to merge duplicates (e.g., if 'Terra' and 'terra' both map to 'terra')
    cleaned = {mapping.get(str(t).strip().lower(), str(t).strip().lower()) for t in tag_list}
    return sorted(list(cleaned))


def apply_agri_mapping_strict(tag_list, mapping):
    if not isinstance(tag_list, list):
        return []

    # ONLY keep the tag if it's a key in our agriculture map
    # Then replace it with the clean value from the map
    cleaned = {mapping[t.strip().lower()] for t in tag_list if t.strip().lower() in mapping}

    return sorted(list(cleaned))





if __name__ == '__main__':
    # Below is for downloading images
    # download_images_and_create_json(input_file_path='GAIA/val_data.json')

    # 1. Load the dataset (Metadata only)
    print("Loading dataset...")
    ds = load_dataset("azavras/GAIA", split="train")
    df = pd.DataFrame(ds)

    # print(df['tag'].explode().unique())
    # print(df[df['tag'][1].any() == "Agriculture"].count())

    # subsphere_definitions = {
    #     "Agriculture/Farming": ["agri", "farm", "crop", "paddy", "cultiv", "pasture", "vineyard", "orchard",
    #                             "plantation", "irrigation"],
    #     "Land Cover/Land Use": ["land use", "land cover", "land monitoring"], # Removed 'urban', 'built-up', 'landscape'
    #     "Vegetation Dynamics": ["vegetation", "canopy", "greenness", "ndvi", "biomass", "leaf"], # Removed 'forest', 'chlorophyll' (aquatic focus)
    #     "Climate Change/Impact": ["environmental impact"], # Removed 'climate', 'global warming', 'carbon', 'emissions' (too broad)
    #     "Water Quality": ["pollution", "sediment"],  # Removed 'turbidity', 'chlorophyll-a', 'algal', 'phytoplankton'
    #     "Drought/Arid Conditions": ["drought", "arid", "dry", "water scarcity", "evaporation"],  # Removed 'desert'
    #     "Surface Hydrology": ["reservoir", "basin"], # Removed 'surface water', 'lake', 'pond' (usually natural/non-agri)
    #     "Precipitation Patterns": ["precipitation", "rainfall", "monsoon"],  # Removed 'snow', 'storm', 'hail'
    #     "Land Management": ["land management", "reclamation"], # Removed 'restoration', 'conservation', 'protected area'
    #     "Land Surface Temperature": ["surface temperature", "lst", "thermal"],  # Removed 'heat', 'urban heat island'
    #     "Seasonal Changes": ["season", "phenology", "harvest", "planting"] # Added 'harvest/planting', removed 'winter/summer/autumn/interannual'
    # }

    # subsphere_definitions = {
    #     "Agriculture/Farming": ["agri", "farm", "crop", "paddy", "cultiv", "pasture", "vineyard", "orchard",
    #                             "plantation", "irrigation", "agricultural", "agricultural burning",
    #                             "agricultural expansion", "agricultural land", "agricultural monitoring",
    #                             "agricultural practices", "agricultural runoff", "aquaculture", "cash crops",
    #                             "cattle ranching", "center-pivot irrigation", "citrus growing", "coconut farming",
    #                             "coffee plantation", "corn fields", "crop analysis", "crop damage", "crop monitoring",
    #                             "crop yield", "croplands", "desert agriculture", "dryland agriculture", "farming",
    #                             "farmland loss", "fish farms", "flooded rice fields", "flower fields", "livestock",
    #                             "olive oil production", "palm oil", "pastoralism", "precision agriculture",
    #                             "rice cultivation", "wheat production"],
    #     "Land Cover/Land Use": ["land use", "land cover", "land monitoring", "agricultural land use",
    #                             "land cover analysis", "land cover change", "land cover classification",
    #                             "land cover mapping", "land use change", "land use and land cover", "landcover",
    #                             "landcover change"],
    #     "Vegetation Dynamics": ["vegetation", "canopy", "greenness", "ndvi", "biomass", "leaf", "ndvi anomaly",
    #                             "vegetation change", "vegetation conditions", "vegetation cover", "vegetation health",
    #                             "vegetation index", "vegetation phenology", "vegetation stress", "leaf area index",
    #                             "plant indices", "photosynthesis"],
    #     "Climate Change/Impact": ["environmental impact", "agricultural impact", "climate adaptation", "climate impact",
    #                               "climate resilience", "climate vulnerability"],
    #     "Water Quality": ["nutrient pollution", "sediment runoff", "agricultural runoff", "eutrophication", "nutrient runoff",
    #                       "water quality", "water pollution"],
    #     "Drought/Arid Conditions": ["drought", "arid", "dry", "water scarcity", "evaporation", "aridity",
    #                                 "desertification", "drought impact", "drought monitoring", "flash drought",
    #                                 "water shortage"],
    #     "Surface Hydrology": ["reservoir", "basin", "water storage", "lakes and reservoirs", "ephemeral water body",
    #                           "water bodies"],
    #     "Precipitation Patterns": ["precipitation", "rainfall", "monsoon", "rainfall patterns", "monsoon rains",
    #                                "wet season", "precipitation analysis"],
    #     "Land Management": ["land management", "reclamation", "soil conservation", "landscape management",
    #                         "conservation efforts", "land administration"],
    #     "Land Surface Temperature": ["surface temperature", "lst", "thermal", "brightness temperature",
    #                                  "thermal imagery", "thermal patterns", "land surface temperature"],
    #     "Seasonal Changes": ["season", "phenology", "harvest", "planting", "growing season", "harvest season",
    #                          "planting season", "vegetation phenology", "wet season", "dry season", "spring vegetation",
    #                          "autumn vegetation"]
    # }
    #
    # tag_to_subsphere = {}
    # all_tags_in_ds = df['tag'].explode().dropna().unique()
    #
    # for tag in all_tags_in_ds:
    #     tag_lower = str(tag).lower()
    #     matched_categories = []
    #     for category, keywords in subsphere_definitions.items():
    #         if any(k in tag_lower for k in keywords):
    #             matched_categories.append(category)
    #     if matched_categories:
    #         tag_to_subsphere[tag] = matched_categories
    #
    #
    # # 3. Create a helper function to identify if a row belongs to our target universe
    # def get_row_subspheres(tags):
    #     if not isinstance(tags, list): return set()
    #     categories = set()
    #     for t in tags:
    #         if t in tag_to_subsphere:
    #             categories.update(tag_to_subsphere[t])
    #     return categories
    #
    #
    # # 4. Apply the mapping to the dataframe
    # df['found_subspheres'] = df['tag'].apply(get_row_subspheres)
    #
    # # 5. Calculate Statistics
    # total_images_in_universe = df[df['found_subspheres'].map(len) > 0]
    #
    # print(f"📊 --- DATASET SUBSPHERE CENSUS ---")
    # print(f"Total Unique Images in these categories: {len(total_images_in_universe)}")
    # print(f"Percentage of Total Dataset: {(len(total_images_in_universe) / len(df)) * 100:.2f}%\n")
    #
    # # 6. Break down by category
    # category_counts = {}
    # for category in subsphere_definitions.keys():
    #     count = df['found_subspheres'].apply(lambda x: category in x).sum()
    #     category_counts[category] = count
    #
    # # Sort and display
    # sorted_counts = dict(sorted(category_counts.items(), key=lambda item: item[1], reverse=True))
    # print(f"{'Subsphere Category':<30} | {'Image Count':<10}")
    # print("-" * 45)
    # for cat, count in sorted_counts.items():
    #     print(f"{cat:<30} | {count:<10}")
    #
    # all_unique_tags = df['tag'].explode().unique()
    # print(f"There are {len(all_unique_tags)} unique tags.")


    # ---------- Resolution, Modalities, Satellite
    cols_to_check = ['resolution', 'modalities', 'satellite']
    for col in cols_to_check:
        print(f"--- {col.upper()} ---")
        print(df[col].explode().value_counts())
        print("\n")

    # ---------- Country
    df['country'] = df['location'].str.split(',').str[-1].str.strip()
    print("Top 10 Countries:")
    print(df['country'].value_counts().head(10))
    print()

    # ---------- Clean set
    full_mapping = {**master_clean_map, **scientific_map}
    df['tag'] = df['tag'].apply(apply_mapping, mapping=full_mapping)

    # df['tag'] = df['tag'].apply(apply_agri_mapping_strict, mapping=agri_specialized_map)

    df['satellite_clean'] = df['satellite'].apply(apply_mapping_robust, mapping=satellite_mapping)

    original_unique = df['satellite'].explode().nunique()
    clean_unique = df['satellite_clean'].explode().nunique()
    print(f"Unique Satellites (Before): {original_unique}")
    print(f"Unique Satellites (After):  {clean_unique}")
    print(f"Reduction: {original_unique - clean_unique} categories merged.")

    df['modalities_clean'] = df['modalities'].apply(apply_mapping_robust, mapping=modality_mapping)

    before_m = df['modalities'].explode().nunique()
    after_m = df['modalities_clean'].explode().nunique()
    print(f"Unique Modalities (Before): {before_m}")
    print(f"Unique Modalities (After):  {after_m}")
    print(f"Reduction: {before_m - after_m} modality types merged.")

    # ---------- Class distribution
    df['earth_sphere'] = df['tag'].apply(classify_to_sphere)
    print(f"--- Distribution of Classes ---")
    print(df['earth_sphere'].value_counts())

    # ---------- Inter-class relations
    sphere_dummies = pd.get_dummies(df['earth_sphere'])
    sphere_corr = sphere_dummies.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(sphere_corr, annot=True, cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5)
    plt.title("Inter-Class Correlation", fontsize=15)
    plt.show()

    df['sphere_count'] = df['tag'].apply(count_spheres)
    complexity_dist = df['sphere_count'].value_counts(normalize=True) * 100
    print(f"\nPercentage of images contained in at least 2 classes: {complexity_dist[2:].sum():.2f}%")

    df['all_spheres'] = df['tag'].apply(get_all_spheres)
    all_pairs = []
    for spheres in df['all_spheres']:
        if len(spheres) >= 2:
            all_pairs.extend(combinations(spheres, 2))
    pair_counts = Counter(all_pairs)
    pair_df = pd.DataFrame(pair_counts.items(), columns=['Pair', 'Count']).sort_values(by='Count', ascending=False)
    pair_df['Pair_Name'] = pair_df['Pair'].apply(lambda x: f"{x[0]} & {x[1]}")

    plt.figure(figsize=(12, 7))
    sns.barplot(data=pair_df.head(10), x='Count', y='Pair_Name', palette='viridis', hue='Pair_Name', legend=False)
    plt.title("Top 10 Inter-Class Interactions", fontsize=16)
    plt.show()

    # ---------- Other metrics for checking tags per image
    tags_per_image = df['tag'].apply(len)
    print(f"\nAverage tags per image: {tags_per_image.mean():.2f}")
    print(f"Max tags on one image: {tags_per_image.max()}")

    plt.figure()
    tags_per_image.value_counts().sort_index().plot(kind='bar')
    plt.title("Distribution of Tag Density per Image")
    plt.xlabel("Number of Tags")
    plt.ylabel("Number of Images")
    plt.show()

    # ---------- Other data interactions
    mlb = MultiLabelBinarizer()
    tag_df = pd.DataFrame(mlb.fit_transform(df['tag']), columns=mlb.classes_)

    tag_counts = tag_df.sum().sort_values(ascending=False)
    print(f"\nTotal unique tags after mapping: {len(tag_counts)}")
    print(f"Tags appearing only once: {(tag_counts == 1).sum()}")
    print(f"Tags appearing in > 100 images: {(tag_counts > 100).sum()}")

    frequent_tags = tag_counts[tag_counts > 100].index.tolist()
    remaining_mask = tag_df[frequent_tags].sum(axis=1) > 0

    original_total = len(df)
    remaining_total = remaining_mask.sum()
    removed_total = original_total - remaining_total

    print(f"\n--- Dataset Size Analysis ---")
    print(f"Original images:         {original_total}")
    print(f"Images with >=1 freq tag: {remaining_total}")
    print(f"Images 'Removed':        {removed_total}")
    print(f"Percentage Kept:         {(remaining_total / original_total) * 100:.2f}%")

    meta_to_exclude = {
        'remote sensing', 'satellite imagery', 'earth observation',
        'false color', 'geography', 'environment'
    }
    final_content_tags = [t for t in frequent_tags if t not in meta_to_exclude]

    remaining_tags = sorted(final_content_tags)

    print()
    print("-" * 30)
    print(f"FINAL COUNT: {len(remaining_tags)} highly relevant"
          f" tags")
    print(f"Sample of remaining tags: {remaining_tags[:10]}...")

    # ---------- Correlation heatmap
    correlation_matrix = tag_df[remaining_tags].corr()

    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Keep just lower-left corner

    plot_df = correlation_matrix.rename(columns=lambda x: x.replace('_', ' ').title())
    plot_df.index = [x.replace('_', ' ').title() for x in plot_df.index]

    sns.heatmap(plot_df,
                mask=mask,
                cmap='RdBu_r',  # Red for positive, Blue for negative
                center=0,
                # vmax=0.25,
                # vmin=-0.25,
                annot=len(remaining_tags) < 30,  # Only if you have < 20-25 tags
                fmt=".2f",
                linewidths=.5,
                cbar_kws={"shrink": .7, "label": "Correlation Coefficient"})

    plt.title("Correlation of Scientific GAIA Tags", fontsize=18, pad=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # ---------- Tags that interact with each other
    check_tag_context(tag_df, 'wildfire')

    # ---------- Clustermap
    # Clustermaps group related physical phenomena (e.g., all 'Ice' tags move together)
    display_matrix = correlation_matrix.rename(
        columns=lambda x: x.replace('_', ' ').title(),
        index=lambda x: x.replace('_', ' ').title()
    )

    g = sns.clustermap(
        display_matrix,
        cmap='RdBu_r',  # Consistent color scheme with the heatmap
        center=0,
        # vmin=-0.25,  # Optional: Adjust these to "zoom" into colors
        # vmax=0.25,  # if your correlations are generally low
        figsize=(18, 18),
        annot=False,
        dendrogram_ratio=0.1,  # Compacts the trees to give the heatmap more room
        cbar_pos=(0.015, 0.89, 0.02, 0.1),  # Moves colorbar to top left (cleaner)
        linewidths=0.5,  # Adds a thin grid line between squares
        linecolor='whitesmoke',
        tree_kws=dict(linewidths=1.25)  # Makes the dendrogram lines thicker and easier to follow
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=10)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=10)
    plt.show()

    # print(df)
    # download_by_tag(df, target_tag='wildfire')

    data_dir = r'\\wsl.localhost\Ubuntu-22.04\home\antonio\projects\mamba-gaia-ir\data\GAIA'
    download_dataset_by_spheres_v3(df, output_dir=data_dir, min_count=100)

    # mapping_results = df['tag'].explode().value_counts()
    # print("--- NEW CATEGORY DISTRIBUTION ---")
    # print(mapping_results)

    # data_dir = r'D:/datasets/GAIA_Agri'
    # download_agri_dataset_by_tag(df, data_dir)

    # df_universe = df[df['found_subspheres'].map(len) > 0].copy()
    # print(f"Total images being sent to downloader: {len(df_universe)}")
    # download_agri_master_dataset(df_universe)

    for sphere in df['earth_sphere'].unique():
        print(f"\n--- SAMPLE TAGS FOR: {sphere} ---")
        # Sample 10 random images from this sphere
        samples = df[df['earth_sphere'] == sphere]['tag'].sample(min(30, len(df))).tolist()
        for i, tags in enumerate(samples):
            print(f"{i + 1}: {tags}")
