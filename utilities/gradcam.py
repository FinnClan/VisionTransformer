# utilities/gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

# utilities/gradcam.py
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model                      # do NOT .eval() here
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        self.activations = out
        out.register_hook(lambda g: setattr(self, "gradients", g))  # tensor-level grad hook

    def generate(self, input_tensor, class_idx=None):
        self.gradients = None;
        self.activations = None
        was_training = self.model.training
        with torch.set_grad_enabled(True):
            self.model.eval()
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = int(output.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            output[:, class_idx].sum().backward()

        if self.gradients is None or self.activations is None:
            h, w = input_tensor.shape[-2:]
            cam_np = np.zeros((h, w), dtype=np.float32)
        else:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [B,C,1,1]
            cam = torch.relu((weights * self.activations).sum(dim=1)) # [B,H,W]
            cam0 = cam[0]
            cam0 = (cam0 - cam0.min()) / (cam0.max() - cam0.min() + 1e-8)
            cam_np = cam0.detach().cpu().float().numpy()

        if was_training: self.model.train()
        return cam_np


def overlay_heatmap_on_image(image_pil, heatmap, alpha=0.30):
    heatmap_resized = cv2.resize(heatmap, image_pil.size)
    heatmap_color = cv2.applyColorMap(
        np.uint8(np.clip(255.0 * heatmap_resized, 0, 255)),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_np = np.array(image_pil).astype(np.float32)
    overlay = img_np * (1.0 - alpha) + heatmap_color.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)
