"""Shared collate function for eval-mode (val/test) datasets that group
multiple captions per image (GAIA, RSITMD, RSICD). Keeps per-image caption
lists flat, in image-major order, instead of letting default_collate
transpose them.
"""

import torch


def eval_collate_fn(batch):
    images, input_ids, attention_mask, captions_lists, idxs = zip(*batch)
    images = torch.stack(images)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    flat_captions = [c for caps in captions_lists for c in caps]
    idxs = torch.tensor(idxs, dtype=torch.long)
    return images, input_ids, attention_mask, flat_captions, idxs