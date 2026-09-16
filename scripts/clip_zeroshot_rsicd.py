# NOTE: Transcribed from three screenshots of the same file, in order.
# The screenshots have gaps between them - missing parts are marked with `...`.

"""Zero-shot CLIP baseline on RSICD - no fine-tuning at all.

Uses off-the-shelf CLIP (both towers, unmodified) directly, to measure
CLIP's actual zero-shot retrieval capability. This is different from the
"CLIP-pretrained transformer text tower" experiment (configs/experiment/
clip_pretrained_text_exp.yaml), which only replaces our own text encoder
while keeping our own GAIA-trained image tower plus randomly-initialized
projection heads - those heads have never been trained, so that pipeline
cannot measure CLIP's real zero-shot alignment. Real CLIP zero-shot needs
CLIP's own image tower and CLIP's own already-aligned joint embedding
space, with no extra heads in between, which is what this script does.

Reuses RSICDDataset's eval_records (grouped image -> captions, built by
CSV loading + os.path.exists() filtering) directly, bypassing its
tokenizer/transform machinery since CLIP needs its own tokenizer and
preprocessing.

Usage:
    python scripts/clip_zeroshot_rsicd.py --data_dir /path/to/data/RSICD --split test
"""

import argparse

import open_clip
import rootutils
import torch
from PIL import Image

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.rsicd_datamodule import RSICDDataset


class _NoOpTokenizer:
    """RSICDDataset's __init__ requires a tokenizer argument, but we only
    read its .eval_records attribute afterward, never call __getitem__ -
    so this stub is never actually invoked."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Not used - this script reads eval_records directly.")


def compute_retrieval_metrics(img_embs, txt_embs, img_ids, txt_img_ids):
    # Docstring partially cut off in screenshot; visible tail:
    # (src/models/mamba3_module.py) so numbers are computed the same way."""
    sim_matrix = img_embs @ txt_embs.t()
    relevant = img_ids.unsqueeze(1) == txt_img_ids.unsqueeze(0)  # [N_img, N_txt]

    results = {}
    for k in [1, 5, 10, 20]:
        kk = min(k, sim_matrix.shape[1])
        top_k_i2t = sim_matrix.topk(kk, dim=1).indices
        results[f"I2T_R{k}"] = (
            relevant.gather(1, top_k_i2t).any(dim=1).float().mean().item()
        )

        sim_t = sim_matrix.t()
        kk_t = min(k, sim_t.shape[1])
        top_k_t2i = sim_t.topk(kk_t, dim=1).indices
        candidate_ids = img_ids[top_k_t2i]
        owning_ids = txt_img_ids.unsqueeze(1)
        results[f"T2I_R{k}"] = (
            (candidate_ids == owning_ids).any(dim=1).float().mean().item()
        )

    results["mean_R1"] = 0.5 * (results["I2T_R1"] + results["T2I_R1"])
    return results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to data/RSICD")
    parser.add_argument(
        "--split", default="test", choices=["test", "val", "combined_val"]
    )
    parser.add_argument("--model_name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print(f"Loading CLIP {args.model_name} ({args.pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained
    )

    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(args.device).eval()

    dataset = RSICDDataset(
        root_dir=args.data_dir,
        tokenizer=_NoOpTokenizer(),
        transform=None,
        split=args.split,
    )

    records = dataset.eval_records  # List[Tuple[img_path, List[str]]]
    print(f"Evaluating on {len(records)} images.")

    img_embs, img_ids = [], []
    txt_embs, txt_img_ids = [], []

    for img_id, (img_path, captions) in enumerate(records):
        image = Image.open(img_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(args.device)
        img_emb = model.encode_image(image_input)
        img_embs.append(img_emb.cpu())
        img_ids.append(img_id)

        for caption in captions:
            tokens = tokenizer([caption]).to(args.device)
            txt_emb = model.encode_text(tokens)
            txt_embs.append(txt_emb.cpu())
            txt_img_ids.append(img_id)

        if img_id % 100 == 0:
            print(f"\rProcessed {img_id}/{len(records)} images")

    img_embs = torch.cat(img_embs)
    txt_embs = torch.cat(txt_embs)
    img_ids_t = torch.tensor(img_ids)
    txt_img_ids_t = torch.tensor(txt_img_ids)

    img_embs = torch.nn.functional.normalize(img_embs, p=2, dim=-1)
    txt_embs = torch.nn.functional.normalize(txt_embs, p=2, dim=-1)

    results = compute_retrieval_metrics(img_embs, txt_embs, img_ids_t, txt_img_ids_t)
    
    print("\n=== CLIP zero-shot RSICD results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
if __name__ == "__main__":
    main()