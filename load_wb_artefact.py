import json
from pathlib import Path

import torch
import wandb

entity = "mihail-chirtu-polytechnica-university-of-bucharest"
project = "lightning-hydra-template"
run_id = "YOUR_RUN_ID"

api = wandb.Api()

artifact = api.artifact(
    f"{entity}/{project}/retrieval-embeddings:test-run-{run_id}"
)

artifact_dir = Path(artifact.download())

tensors = torch.load(
    artifact_dir / "embeddings.pt",
    map_location="cpu",
    weights_only=True,
)

with open(artifact_dir / "index.json", encoding="utf-8") as file:
    index = json.load(file)

image_embeddings = tensors["image_embeddings"]
text_embeddings = tensors["text_embeddings"]
valid_text_mask = tensors["text_valid_mask"]

# Usually exclude padded caption slots from analysis.
valid_text_embeddings = text_embeddings[valid_text_mask]
valid_text_ids = [
    text_id
    for text_id, valid in zip(
        index["text_ids"],
        valid_text_mask.tolist(),
        strict=True,
    )
    if valid
]

print(image_embeddings.shape)
print(valid_text_embeddings.shape)



import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Keep plotting manageable if there are many captions.
max_text_points = 5_000
valid_text_embeddings = valid_text_embeddings[:max_text_points]

combined = torch.cat(
    [image_embeddings, valid_text_embeddings],
    dim=0,
).numpy()

projection = PCA(n_components=2).fit_transform(combined)

num_images = image_embeddings.shape[0]
image_xy = projection[:num_images]
text_xy = projection[num_images:]

plt.figure(figsize=(10, 8))

plt.scatter(
    image_xy[:, 0],
    image_xy[:, 1],
    s=20,
    alpha=0.8,
    label="Images",
)

plt.scatter(
    text_xy[:, 0],
    text_xy[:, 1],
    s=10,
    alpha=0.35,
    label="Captions",
)

plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.title(f"Joint retrieval embedding space — run {run_id}")
plt.legend()
plt.tight_layout()
plt.show()