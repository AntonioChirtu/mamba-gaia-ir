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


def download_dataset_by_spheres(df, spheres_dict, output_dir='GAIA/', limit_per_tag=100):
    """
    Downloads images organized by Earth Sphere and Tag.
    """
    for sphere, tags in spheres_dict.items():
        print(f"\n=== Processing Sphere: {sphere} ===")

        for tag in tags:
            # 1. Filter for the tag
            mask = df['tag'].apply(lambda x: tag in x if isinstance(x, list) else False)
            filtered_df = df[mask].head(limit_per_tag).copy()

            if filtered_df.empty:
                continue

            # 2. Setup Hierarchical Directories
            # Path: GAIA/val/Biosphere/wildfire/images/
            tag_dir = os.path.join(output_dir, sphere, tag)
            image_dir = os.path.join(tag_dir, 'images')
            os.makedirs(image_dir, exist_ok=True)

            new_json_structure = []
            print(f"Downloading up to {limit_per_tag} images for [{tag}]...")

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

earth_spheres = {
    'Atmosphere': [
        'atmosphere', 'tropical_cyclone', 'dust_storm', 'air_pollution',
        'precipitation', 'smoke_plume', 'climate'
    ],
    'Hydrosphere': [
        'water', 'floods', 'hydrology', 'oceanography', 'phytoplankton',
        'coastal', 'sediment'
    ],
    'Biosphere': [
        'vegetation', 'deforestation', 'wildfire', 'agriculture',
        'land_use', 'drought'
    ],
    'Geosphere': [
        'geology', 'topography', 'desert', 'volcanic_activity',
        'land_management', 'urban', 'natural_disaster'
    ],
    'Cryosphere': [
        'cryosphere', 'polar_regions', 'glacier', 'iceberg',
        'snow', 'winter', 'alaska', 'siberia'
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
    # Calculate overlap for each Earth Sphere
    scores = {sphere: len(set(image_tags) & set(sphere_tags))
              for sphere, sphere_tags in earth_spheres.items()}

    # Pick the sphere with the most matching tags
    top_sphere = max(scores, key=scores.get)

    # Label as 'General/Multi-Sphere' if no specific tags match or if there's a tie
    if scores[top_sphere] == 0:
        return 'General Earth'
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


if __name__ == '__main__':
    # Below is for downloading images
    # download_images_and_create_json(input_file_path='GAIA/val_data.json')

    # 1. Load the dataset (Metadata only)
    print("Loading dataset...")
    ds = load_dataset("azavras/GAIA", split="train")
    df = pd.DataFrame(ds)

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
    download_dataset_by_spheres(df, earth_spheres)
