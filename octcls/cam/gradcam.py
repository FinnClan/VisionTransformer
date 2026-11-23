import torch

from octcls.cam.vit_gradcam import ViTGradCAM
from octcls.cam.gradcam_cnn import GradCAM as _GradCAM
import torch.nn as nn


def _pick_target_layer(model: nn.Module):
    # heuristic: last Conv2d with kernel != 1x1 in the backbone
    if hasattr(model, "cnn"):
        root = model.cnn
    elif hasattr(model, "features"):
        root = model.features
    else:
        root = model
    convs = [m for m in root.modules() if isinstance(m, nn.Conv2d)]
    for m in reversed(convs):
        if getattr(m, "kernel_size", None) != (1, 1):
            return m
    return convs[-1] if convs else None


def make_cam(model: nn.Module):
    """Return a callable CAM extractor for CNN or ViT models."""
    device = next(model.parameters()).device
    # If model exposes a target layer getter, use it
    if hasattr(model, "get_cam_target_layer") and callable(getattr(model, "get_cam_target_layer")):
        layer = model.get_cam_target_layer()
        if layer is not None:
            return _GradCAM(model, layer)
    if hasattr(model, "vit"):
        return ViTGradCAM(model.vit, device=device)
    tl = _pick_target_layer(model if not hasattr(model, "cnn") else model.cnn)
    if tl is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target")
    return _GradCAM(model, tl)
