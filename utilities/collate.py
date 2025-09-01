import torch


def collate_with_meta(batch):
    """
    Collate function for samples of the form (image_tensor, label, meta_dict).
    - Stacks images [B,C,H,W]
    - Stacks labels [B]
    - Aggregates meta into a dict of lists without attempting to stack variable-length fields
    """
    images, labels, metas = zip(*batch)  # lists of length B

    images = torch.stack(images, dim=0)
    labels = torch.as_tensor(labels, dtype=torch.long)

    meta_out = {}
    if metas and isinstance(metas[0], dict):
        keys = set()
        for m in metas:
            keys.update(m.keys())
        for k in keys:
            meta_out[k] = [m.get(k, None) for m in metas]
    else:
        # fall back to raw metas list
        meta_out = list(metas)

    return images, labels, meta_out

