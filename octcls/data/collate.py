import torch


def collate_with_meta(batch):
    images, labels, metas = zip(*batch)
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
        meta_out = list(metas)
    return images, labels, meta_out

__all__ = ["collate_with_meta"]
