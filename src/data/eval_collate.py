"""Shared collate function for eval-mode (val/test) datasets that group
multiple captions per image (GAIA, RSITMD, RSICD). Keeps per-image caption
lists flat, in image-major order, instead of letting default_collate
transpose them.
"""

import torch


def eval_collate_fn(batch):
    images, image_ids, input_ids, attention_mask, captions_lists = zip(*batch)
    images = torch.stack(images)
    image_ids = torch.as_tensor(image_ids, dtype=torch.long)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    flat_captions = [c for caps in captions_lists for c in caps]
    return images, image_ids, input_ids, attention_mask, flat_captions