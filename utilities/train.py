# utilities/train.py
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score
from utilities.gradcam import GradCAM, overlay_heatmap_on_image
from utilities.ViTGradCAM import ViTGradCAM
from contextlib import nullcontext

DATASET_MEAN = [0.5, 0.5, 0.5]
DATASET_STD  = [0.5, 0.5, 0.5]

ONLY_SAVE_WHEN_YOLO = True  # <- set False if you want CAMs even without YOLO


def _as_len_B(x, B, fill=None):
    """Return a plain Python list of length B from x (list/tuple/dict/None/other)."""
    if x is None:
        return [fill] * B
    if isinstance(x, (list, tuple)):
        if len(x) == B:
            return list(x)
        y = list(x)[:B]
        if len(y) < B:
            y += [fill] * (B - len(y))
        return y
    # dict-of-lists case handled at callsite; broadcast for anything else
    return [x] * B

def _safe_get(lst, idx, default=None):
    return lst[idx] if (lst is not None and 0 <= idx < len(lst)) else default

def _usable_bb_in_batch(bboxes, W, H):
    """Return (index, bb_224) for the first usable box in the batch, else (None, None)."""
    if not bboxes:
        return None, None
    B = len(bboxes)
    for j in range(B):
        b = bboxes[j]
        if b is None or (isinstance(b, (list, tuple)) and len(b) >= 4 and all(v == -1 for v in b[:4])):
            continue
        bb = _sanitize_box_xyxy(b, W, H)
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        if (x2 - x1) > 2 and (y2 - y1) > 2:
            return j, bb
    return None, None

def _is_no_box_sentinel(b):
    return (
        b is None or
        (isinstance(b, (list, tuple)) and len(b) >= 4 and all(v == -1 for v in b[:4]))
    )

def _upscale_image_and_box(pil_img, box_xyxy, scale=3):
    W, H = pil_img.size
    big = pil_img.resize((W*scale, H*scale), resample=Image.NEAREST)
    if box_xyxy is None:
        return big, None
    x1, y1, x2, y2 = box_xyxy
    return big, (x1*scale, y1*scale, x2*scale, y2*scale)

def _draw_bbox_styled(img, box, thick=4, bracket=18):
    if box is None:
        return img
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = map(int, box)
    for w, color in ((thick+2, (255,255,255)), (thick, (0,0,0))):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=w)

    def bracket_lines(xx1, yy1, xx2, yy2):
        return [
            ((xx1, yy1), (xx1+bracket, yy1)),
            ((xx1, yy1), (xx1, yy1+bracket)),
            ((xx2, yy1), (xx2-bracket, yy1)),
            ((xx2, yy1), (xx2, yy1+bracket)),
            ((xx1, yy2), (xx1+bracket, yy2)),
            ((xx1, yy2), (xx1, yy2-bracket)),
            ((xx2, yy2), (xx2-bracket, yy2)),
            ((xx2, yy2), (xx2, yy2-bracket)),
        ]
    for w, color in ((thick+2, (255,255,255)), (thick, (0,0,0))):
        for a, b in bracket_lines(x1, y1, x2, y2):
            draw.line([a, b], fill=color, width=w)
    return img

def _flatten4(box):
    if box is None:
        return None
    try:
        import torch as _torch, numpy as _np
        if isinstance(box, _torch.Tensor):
            vals = box.detach().cpu().reshape(-1).tolist()
        elif isinstance(box, _np.ndarray):
            vals = box.reshape(-1).tolist()
        elif isinstance(box, (list, tuple)):
            if len(box) > 0 and isinstance(box[0], (list, tuple)):
                vals = list(box[0])
            else:
                vals = list(box)
        else:
            return None
        return vals[:4] if len(vals) >= 4 else None
    except Exception:
        return None

def _sanitize_box_xyxy(box, img_w, img_h):
    vals = _flatten4(box)
    if vals is None:
        return None
    x1, y1, x2, y2 = [int(round(v)) for v in vals]
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    y2 = max(0, min(y2, img_h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def _as_xyxy4(box):
    if box is None:
        return None
    try:
        import torch as _torch, numpy as _np
        if isinstance(box, _torch.Tensor):
            flat = box.detach().cpu().reshape(-1).tolist()
        elif isinstance(box, _np.ndarray):
            flat = box.reshape(-1).tolist()
        elif isinstance(box, (list, tuple)):
            if len(box) > 0 and isinstance(box[0], (list, tuple)) and len(box[0]) >= 4:
                flat = list(box[0])
            else:
                flat = list(box)
        else:
            return None
        if len(flat) < 4:
            return None
        xyxy = tuple(int(round(v)) for v in flat[:4])
        return xyxy
    except Exception:
        return None

def autocast_off_for(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=False)
    elif device.type == "cpu":
        try:
            return torch.cpu.amp.autocast(enabled=False)
        except AttributeError:
            return nullcontext()
    else:
        return nullcontext()

def pick_target_layer(model: nn.Module):
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
    model_device = next(model.parameters()).device
    if hasattr(model, "vit"):
        return ViTGradCAM(model.vit, device=model_device)
    target_layer = pick_target_layer(model if not hasattr(model, "cnn") else model.cnn)
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return GradCAM(model, target_layer)

def tensor_to_pil(t, mean, std):
    x = t.detach().cpu().clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        x[c] = x[c] * s + m
    x = x.clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(x)

def _crop_and_resize_tensor(img_chw, box_xyxy, out_hw=(224, 224)):
    """img_chw: [C,H,W] tensor. box_xyxy can be list/tuple/ndarray/tensor of 4 values or nested.
    Returns [C,out_h,out_w] crop resized. Falls back to full image if invalid.
    """
    C, H, W = img_chw.shape
    try:
        bb = _sanitize_box_xyxy(box_xyxy, img_w=W, img_h=H)
    except Exception:
        bb = None
    if bb is not None:
        x1, y1, x2, y2 = bb
        if x2 > x1 + 1 and y2 > y1 + 1:
            crop = img_chw[:, y1:y2, x1:x2]
            return F.interpolate(crop.unsqueeze(0), size=out_hw, mode='bilinear', align_corners=False)[0]
    # fallback
    return F.interpolate(img_chw.unsqueeze(0), size=out_hw, mode='bilinear', align_corners=False)[0]


def train_model(device, model, train_loader, epochs=50, heatmap_every=10, save_dir="cam_outputs", roi_agg: str = "mean"):
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.to(device)
    model.train()

    cam = make_cam(model)

    loss_history, acc_history = [], []

    for epoch in range(epochs):
        running_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            # --- normalize batch contents (robust to dict-of-lists / list-of-dicts / legacy) ---
            images, labels = batch[0], batch[1]
            B = images.size(0)

            paths, bboxes, bboxes_k = None, None, None
            if len(batch) >= 3:
                meta = batch[2]
                if isinstance(meta, dict):  # dict-of-lists post-collate
                    paths  = _as_len_B(meta.get("path", None), B, fill=None)
                    bboxes = _as_len_B(meta.get("bbox_224", None), B, fill=None)
                    bboxes_k = _as_len_B(meta.get("bboxes_224", None), B, fill=None)
                elif isinstance(meta, (list, tuple)) and len(meta) > 0 and isinstance(meta[0], dict):
                    paths  = _as_len_B([m.get("path", None) for m in meta], B, fill=None)
                    bboxes = _as_len_B([m.get("bbox_224", None) for m in meta], B, fill=None)
                    bboxes_k = _as_len_B([m.get("bboxes_224", None) for m in meta], B, fill=None)
                else:
                    # legacy forms
                    paths  = _as_len_B(meta, B, fill=None)
                    bboxes = _as_len_B(batch[3] if len(batch) >= 4 else None, B, fill=None)
                    bboxes_k = [None] * B
            else:
                paths  = [None] * B
                bboxes = [None] * B
                bboxes_k = [None] * B
            # ------------------------------------------------------------------------------------

            images, labels = images.to(device), labels.to(device)

            # --- standard training step ---
            optimizer.zero_grad(set_to_none=True)
            # Aggregate logits over top-K YOLO crops per image if provided
            agg_logits = []
            for j in range(B):
                img_j = images[j]
                # Collect bboxes for this item
                bb_224 = None
                bb_list = None
                if len(batch) >= 3:
                    meta = batch[2]
                    if isinstance(meta, dict):
                        bb_list = _safe_get(meta.get("bboxes_224", None), j, None)
                        bb_224 = _safe_get(meta.get("bbox_224", None), j, None)
                    elif isinstance(meta, (list, tuple)) and len(meta) > j and isinstance(meta[j], dict):
                        bb_list = meta[j].get("bboxes_224", None)
                        bb_224 = meta[j].get("bbox_224", None)
                # Normalize forms
                if bb_list is None or (isinstance(bb_list, (list, tuple)) and len(bb_list) == 0):
                    bb_list = []
                    if bb_224 is not None and not _is_no_box_sentinel(bb_224):
                        bb_list = [bb_224]

                # Build crops
                crops = []
                if isinstance(bb_list, (list, tuple)) and len(bb_list) > 0:
                    for bb in bb_list:
                        if bb is None or (isinstance(bb, (list, tuple)) and len(bb) >= 4 and all(v == -1 for v in bb[:4])):
                            continue
                        crop = _crop_and_resize_tensor(img_j, bb, out_hw=(224, 224))
                        crops.append(crop)
                if not crops:
                    crops = [_crop_and_resize_tensor(img_j, None, out_hw=(224, 224))]

                crops_t = torch.stack(crops, dim=0).to(device)  # [K,C,224,224]
                logits_k = model(crops_t)                        # [K,num_classes]
                if roi_agg == "max":
                    logits = logits_k.max(dim=0, keepdim=False).values
                else:
                    logits = logits_k.mean(dim=0, keepdim=False)
                agg_logits.append(logits)

            outputs = torch.stack(agg_logits, dim=0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # ------------------------------

            # --- Grad-CAM visualization at a batch frequency ---
            if heatmap_every and heatmap_every > 0 and (batch_idx % heatmap_every == 0):
                W, H = 224, 224  # expected overlay size (matches your transforms)
                was_training = model.training
                model.eval()
                torch.set_grad_enabled(True)
                with autocast_off_for(device):
                    for j in range(B):
                        # collect all valid boxes for display
                        disp_boxes = []
                        k_list = _safe_get(bboxes_k, j, None)
                        if isinstance(k_list, (list, tuple)) and len(k_list) > 0:
                            for bb in k_list:
                                if bb is None or _is_no_box_sentinel(bb):
                                    continue
                                bbv = _sanitize_box_xyxy(bb, W, H)
                                if bbv is not None:
                                    disp_boxes.append(bbv)
                        else:
                            b_j = _safe_get(bboxes, j, None)
                            bb_224 = _sanitize_box_xyxy(b_j, W, H) if (b_j is not None and not _is_no_box_sentinel(b_j)) else None
                            if bb_224 is not None:
                                disp_boxes.append(bb_224)
                        if ONLY_SAVE_WHEN_YOLO and not disp_boxes:
                            continue

                        img_j = images[j:j+1]  # [1,C,H,W]
                        cam_map = cam(img_j, class_idx=None)  # or class_idx=int(labels[j].item())

                        vis_image = tensor_to_pil(images[j], DATASET_MEAN, DATASET_STD)
                        overlay = overlay_heatmap_on_image(vis_image, cam_map, alpha=0.30)

                        overlay_big = overlay.resize((512, 512), resample=Image.NEAREST)
                        if disp_boxes:
                            sx = 512 / W; sy = 512 / H
                            for bb_224 in disp_boxes:
                                x1, y1, x2, y2 = bb_224
                                x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                                if (x2 - x1) > 2 and (y2 - y1) > 2:
                                    overlay_big = _draw_bbox_styled(overlay_big, (x1, y1, x2, y2), thick=4, bracket=18)
                        elif ONLY_SAVE_WHEN_YOLO:
                            continue

                        base_path = _safe_get(paths, j, None)
                        base = (os.path.splitext(os.path.basename(base_path))[0]
                                if base_path else f"epoch{epoch:02d}_batch{batch_idx:04d}_idx{j}")
                        overlay_big.save(os.path.join(save_dir, f"{base}_cam.png"))
                if was_training:
                    model.train()
            # -----------------------------------------------------------------------------------

            # metrics
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            running_loss += loss.item()

        acc = accuracy_score(all_labels, all_preds)
        avg_loss = running_loss / max(1, len(train_loader))
        loss_history.append(avg_loss)
        acc_history.append(acc)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")

    # plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(loss_history, label='Loss'); plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(acc_history, label='Accuracy'); plt.title("Training Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True)
    plt.tight_layout(); plt.savefig("training_curves.png"); plt.show()

    # Clean up CAM hooks
    try:
        cam.release()
    except Exception:
        pass

    return model
