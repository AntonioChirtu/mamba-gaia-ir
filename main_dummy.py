import json
import os

# Pick one of the paths from your logs
test_json = "/home/antonio/projects/mamba-gaia-ir/data/GAIA_tiles/Atmosphere/air_pollution/metadata.json"
test_dir = os.path.dirname(test_json)

with open(test_json, 'r') as f:
    data = json.load(f)
    first_rel_path = data[0]['image_path']
    full_path = os.path.join(test_dir, first_rel_path)

    print(f"JSON says relative path is: {first_rel_path}")
    print(f"Python is looking for: {full_path}")
    print(f"Does it exist? {os.path.exists(full_path)}")
    print(f"Actual files in that folder: {os.listdir(test_dir)[:5]}")