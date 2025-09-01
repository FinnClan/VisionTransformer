# gradcam_vit.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

class ViTGradCAM:
    """
    Classic Grad-CAM on torchvision ViT by targeting vit.conv_proj (patch embedding conv).
    Produces a HxW heatmap for a single image tensor [1,3,224,224].
    """
    def __init__(self, vit_model, target_layer=None, device='cpu'):
        self.model = vit_model
        if device is None:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # pick target layer (default: vit.conv_proj)
        # inside ViTGradCAM.__init__
        if target_layer is not None:
            self.target_layer = target_layer
        else:
            # Try common names across libraries
            # torchvision: model.conv_proj
            # timm: model.patch_embed.proj
            if hasattr(self.model, "conv_proj"):
                self.target_layer = self.model.conv_proj
            elif hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "proj"):
                self.target_layer = self.model.patch_embed.proj
            else:
                raise AttributeError(
                    "Could not find a patch-embedding conv layer. "
                    "Pass `target_layer=` explicitly (e.g., vit.conv_proj or vit.patch_embed.proj)."
                )

        self.activations = None
        self.gradients = None

        # forward & backward hooks
        self.fwd_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        # out: [B, C, 14, 14]
        self.activations = out.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output[0]: grad wrt output of target layer -> [B, C, 14, 14]
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, class_idx=None):
        """
        image_tensor: [1,3,224,224] (normalized same as training)
        class_idx: optional int. If None, uses predicted class.
        returns: heatmap (H, W) in [0,1], numpy array
        """
        self.model.zero_grad()
        logits = self.model(image_tensor)  # [1,num_classes]

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]          # scalar
        score.backward(retain_graph=True)

        # activations & gradients: [1, C, 14, 14]
        A = self.activations           # features
        dA = self.gradients            # grads

        # weights: channel-wise global-average of gradients
        weights = dA.mean(dim=(2, 3), keepdim=True)      # [1, C, 1, 1]
        cam = (weights * A).sum(dim=1, keepdim=False)    # [1, 14, 14]
        cam = F.relu(cam)                                # clamp negatives
        cam = cam[0]                                     # [14, 14]

        # normalize to [0,1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # upsample to 224x224
        cam = cam.unsqueeze(0).unsqueeze(0)              # [1,1,14,14]
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam[0, 0].cpu().numpy()                    # (224, 224)
        return cam

    def release(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def denormalize(img_tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """ img_tensor: [3,H,W] -> [3,H,W] de-normalized in [0,1] """
    x = img_tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        x[c] = x[c] * s + m
    return x.clamp(0, 1)

def overlay_heatmap_on_image(img_chw, heatmap, alpha=0.30):
    """
    img_chw: [3,H,W] tensor in [0,1] (de-normalized)
    heatmap: (H,W) numpy array in [0,1]
    returns PIL.Image with jet heatmap overlay
    """
    import matplotlib.cm as cm
    img = (img_chw.permute(1,2,0).numpy() * 255).astype(np.uint8)     # HWC
    color_map = cm.get_cmap('jet')
    heat_rgba = color_map(heatmap)            # HxWx4
    heat_rgb = (heat_rgba[..., :3] * 255).astype(np.uint8)

    overlay = (alpha * heat_rgb + (1 - alpha) * img).astype(np.uint8)
    return Image.fromarray(overlay)
