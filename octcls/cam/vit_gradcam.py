from __future__ import annotations

import torch
import torch.nn.functional as F


class ViTGradCAM:
    def __init__(self, vit_model, target_layer=None, device='cpu'):
        self.model = vit_model
        if device is None:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        if target_layer is not None:
            self.target_layer = target_layer
        else:
            if hasattr(self.model, "conv_proj"):
                self.target_layer = self.model.conv_proj
            elif hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "proj"):
                self.target_layer = self.model.patch_embed.proj
            else:
                raise AttributeError("Could not find a patch-embedding conv layer.")

        self.activations = None
        self.gradients = None
        self.fwd_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        A = self.activations
        dA = self.gradients
        weights = dA.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1)
        cam = F.relu(cam)[0]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return cam[0, 0].cpu().numpy()

    def release(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

