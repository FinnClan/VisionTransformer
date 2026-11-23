from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM++ style attribution for convolutional backbones."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        try:
            self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)
        except AttributeError:
            self.bwd_handle = target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.gradients = None
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
            cam = torch.zeros((1, h, w), device=input_tensor.device)
        else:
            grads = self.gradients
            acts = self.activations

            grads_2 = grads.pow(2.0)
            grads_3 = grads_2 * grads
            # Equation 7 from Grad-CAM++: per-location alpha weights
            denom = 2.0 * grads_2 + (acts * grads_3).sum(dim=(2, 3), keepdim=True)
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
            alpha = grads_2 / (denom + 1e-8)

            positive_grads = torch.relu(grads)
            weights = (alpha * positive_grads).sum(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=1, keepdim=False))

        cam0 = cam[0]
        cam0 = (cam0 - cam0.min()) / (cam0.max() - cam0.min() + 1e-8)
        cam0 = F.interpolate(cam0.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        if was_training:
            self.model.train()
        return cam0.detach().cpu().numpy()

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def release(self):
        try:
            self.fwd_handle.remove()
        except Exception:
            pass
        try:
            self.bwd_handle.remove()
        except Exception:
            pass
