from __future__ import annotations

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score
from contextlib import nullcontext

from octcls.cam.gradcam import make_cam
from octcls.viz.overlay import overlay_heatmap_on_image
from octcls.utils.logging import get_logger

DATASET_MEAN = [0.5, 0.5, 0.5]
DATASET_STD  = [0.5, 0.5, 0.5]

ONLY_SAVE_WHEN_YOLO = True
logger = get_logger(__name__)


def _as_len_B(x, B, fill=None):
    if x is None:
        return [fill] * B
    if isinstance(x, (list, tuple)):
        if len(x) == B:
            return list(x)
        y = list(x)[:B]
        if len(y) < B:
            y += [fill] * (B - len(y))
        return y
    return [x] * B


def _safe_get(lst, idx, default=None):
    return lst[idx] if (lst is not None and 0 <= idx < len(lst)) else default


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


def _is_no_box_sentinel(b):
    return (
        b is None or
        (isinstance(b, (list, tuple)) and len(b) >= 4 and all(v == -1 for v in b[:4]))
    )


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


def tensor_to_pil(t, mean, std):
    x = t.detach().cpu().clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        x[c] = x[c] * s + m
    x = x.clamp(0, 1)
    x = (x.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(x)


def _crop_and_resize_tensor(img_chw, box_xyxy, out_hw=(224, 224)):
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
    return F.interpolate(img_chw.unsqueeze(0), size=out_hw, mode='bilinear', align_corners=False)[0]


def train_model(device, model, train_loader, epochs=50, heatmap_every=10, save_dir="cam_outputs",
                roi_agg: str = "mean", val_loader=None, save_best_path: str = None,
                include_global: bool = True, max_overlay_box_frac: float = 0.95):
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.to(device)
    model.train()

    loss_history, acc_history, val_acc_history = [], [], []
    best_val_acc = -1.0

    for epoch in range(epochs):
        running_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            images, labels = batch[0], batch[1]
            B = images.size(0)

            paths, bboxes, bboxes_k, has_yolo_flags = None, None, None, None
            if len(batch) >= 3:
                meta = batch[2]
                if isinstance(meta, dict):
                    paths  = _as_len_B(meta.get("path", None), B, fill=None)
                    bboxes = _as_len_B(meta.get("bbox_224", None), B, fill=None)
                    bboxes_k = _as_len_B(meta.get("bboxes_224", None), B, fill=None)
                    has_yolo_flags = _as_len_B(meta.get("has_yolo", None), B, fill=0)
                elif isinstance(meta, (list, tuple)) and len(meta) > 0 and isinstance(meta[0], dict):
                    paths  = _as_len_B([m.get("path", None) for m in meta], B, fill=None)
                    bboxes = _as_len_B([m.get("bbox_224", None) for m in meta], B, fill=None)
                    bboxes_k = _as_len_B([m.get("bboxes_224", None) for m in meta], B, fill=None)
                    has_yolo_flags = _as_len_B([m.get("has_yolo", None) for m in meta], B, fill=0)
                else:
                    paths  = _as_len_B(meta, B, fill=None)
                    bboxes = _as_len_B(batch[3] if len(batch) >= 4 else None, B, fill=None)
                    bboxes_k = [None] * B
                    has_yolo_flags = [0] * B
            else:
                paths  = [None] * B
                bboxes = [None] * B
                bboxes_k = [None] * B
                has_yolo_flags = [0] * B

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            # Flatten all ROI crops across the batch for one forward pass
            all_crops = []
            counts = []
            for j in range(B):
                img_j = images[j]
                bb_list = None; bb_224 = None
                if len(batch) >= 3:
                    meta = batch[2]
                    if isinstance(meta, dict):
                        bb_list = _safe_get(meta.get("bboxes_224", None), j, None)
                        bb_224 = _safe_get(meta.get("bbox_224", None), j, None)
                    elif isinstance(meta, (list, tuple)) and len(meta) > j and isinstance(meta[j], dict):
                        bb_list = meta[j].get("bboxes_224", None)
                        bb_224 = meta[j].get("bbox_224", None)
                if not bb_list or (isinstance(bb_list, (list, tuple)) and len(bb_list) == 0):
                    bb_list = []
                    if bb_224 is not None and not _is_no_box_sentinel(bb_224):
                        bb_list = [bb_224]
                # Always include global full image crop if requested
                if include_global:
                    bb_list = list(bb_list) + [None]
                crops = []
                if isinstance(bb_list, (list, tuple)) and len(bb_list) > 0:
                    for bb in bb_list:
                        if bb is None:
                            # Global context crop
                            crops.append(_crop_and_resize_tensor(img_j, None, out_hw=(224, 224)))
                            continue
                        if _is_no_box_sentinel(bb):
                            continue
                        crops.append(_crop_and_resize_tensor(img_j, bb, out_hw=(224, 224)))
                if not crops:
                    crops = [_crop_and_resize_tensor(img_j, None, out_hw=(224, 224))]
                counts.append(len(crops))
                all_crops.extend(crops)

            all_crops_t = torch.stack(all_crops, dim=0).to(device)
            all_logits = model(all_crops_t)
            agg_logits = []
            idx0 = 0
            for c in counts:
                logits_k = all_logits[idx0:idx0+c]
                if roi_agg == "max":
                    logits = logits_k.max(dim=0, keepdim=False).values
                else:
                    logits = logits_k.mean(dim=0, keepdim=False)
                agg_logits.append(logits)
                idx0 += c
            outputs = torch.stack(agg_logits, dim=0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Generate CAM overlays only in the last 5 epochs (or fewer if epochs < 5),
            # and still respect the batch frequency `heatmap_every`.
            last_n = 5
            in_last_window = (epoch >= max(0, epochs - last_n))
            if in_last_window and heatmap_every and heatmap_every > 0 and (batch_idx % heatmap_every == 0):
                W, H = 224, 224
                was_training = model.training
                model.eval()
                torch.set_grad_enabled(True)
                # Create CAM extractor only for this visualization window to avoid hook side-effects during backprop
                cam_extractor = make_cam(model)
                with autocast_off_for(device):
                    for j in range(B):
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

                        has_yolo = bool(_safe_get(has_yolo_flags, j, 0))
                        if (has_yolo and max_overlay_box_frac is not None
                                and len(disp_boxes) == 1 and max_overlay_box_frac > 0):
                            box = disp_boxes[0]
                            area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
                            full = float(W * H)
                            if full > 0 and (area / full) >= max_overlay_box_frac:
                                continue

                        # Crop-aware CAM: select the most influential crop for this image and
                        # compute CAM on that crop, then paste it back into the 224x224 canvas.
                        import torch as _torch
                        import numpy as _np
                        import cv2 as _cv2
                        # Build bb_list consistent with forward path (include global as None)
                        bb_list = []
                        if isinstance(k_list, (list, tuple)) and len(k_list) > 0:
                            for bb in k_list:
                                if bb is None or _is_no_box_sentinel(bb):
                                    continue
                                bbv = _sanitize_box_xyxy(bb, W, H)
                                if bbv is not None:
                                    bb_list.append(bbv)
                        else:
                            if bb_224 is not None:
                                bb_list.append(bb_224)
                        bb_list = list(bb_list) + [None]

                        # Choose class and most influential crop (highest logit for predicted class)
                        class_idx = int(_torch.argmax(outputs[j]).item())
                        start = 0
                        for idx in range(j):
                            start += counts[idx]
                        c_j = counts[j]
                        logits_k = all_logits[start:start + c_j]
                        top_idx = len(bb_list) - 1 if logits_k.shape[0] != len(bb_list) else int(_torch.argmax(logits_k[:, class_idx]).item())
                        bb_sel = bb_list[top_idx] if 0 <= top_idx < len(bb_list) else None

                        crop_in = _crop_and_resize_tensor(images[j], bb_sel, out_hw=(224, 224)).unsqueeze(0)
                        cam_map_crop = cam_extractor(crop_in, class_idx=class_idx)  # numpy (224,224)
                        full_cam = _np.zeros((224, 224), dtype=_np.float32)
                        if bb_sel is None:
                            full_cam = cam_map_crop.astype(_np.float32)
                        else:
                            x1, y1, x2, y2 = map(int, bb_sel)
                            h = max(1, y2 - y1); w = max(1, x2 - x1)
                            cam_resized = _cv2.resize(cam_map_crop, (w, h), interpolation=_cv2.INTER_LINEAR)
                            full_cam[y1:y2, x1:x2] = cam_resized.astype(_np.float32)

                        vis_image = tensor_to_pil(images[j], DATASET_MEAN, DATASET_STD)
                        overlay = overlay_heatmap_on_image(vis_image, full_cam, alpha=0.30)

                        overlay_big = overlay.resize((512, 512), resample=Image.NEAREST)
                        if disp_boxes:
                            sx = 512 / W; sy = 512 / H
                            for bb_224 in disp_boxes:
                                x1, y1, x2, y2 = bb_224
                                x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                                if (x2 - x1) > 2 and (y2 - y1) > 2:
                                    overlay_big = _draw_bbox_styled(overlay_big, (x1, y1, x2, y2), thick=4, bracket=18)

                        base_path = _safe_get(paths, j, None)
                        base = (os.path.splitext(os.path.basename(base_path))[0]
                                if base_path else f"epoch{epoch:02d}_batch{batch_idx:04d}_idx{j}")
                        overlay_big.save(os.path.join(save_dir, f"{base}_cam.png"))
                try:
                    cam_extractor.release()
                except Exception:
                    pass
                if was_training:
                    model.train()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            running_loss += loss.item()

        acc = accuracy_score(all_labels, all_preds)
        avg_loss = running_loss / max(1, len(train_loader))
        loss_history.append(avg_loss)
        acc_history.append(acc)
        msg = f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}"

        if val_loader is not None:
            model.eval()
            all_v_preds, all_v_labels = [], []
            with torch.no_grad():
                for v_batch in val_loader:
                    v_images, v_labels, v_meta = v_batch[0], v_batch[1], (v_batch[2] if len(v_batch) > 2 else None)
                    Bv = v_images.size(0)
                    v_images = v_images.to(device)
                    # Flatten crops across val batch
                    all_crops = []
                    counts = []
                    for j in range(Bv):
                        img_j = v_images[j]
                        bb_list, bb_224 = None, None
                        if isinstance(v_meta, dict):
                            bb_list = _safe_get(v_meta.get("bboxes_224", None), j, None)
                            bb_224 = _safe_get(v_meta.get("bbox_224", None), j, None)
                        elif isinstance(v_meta, (list, tuple)) and len(v_meta) > j and isinstance(v_meta[j], dict):
                            bb_list = v_meta[j].get("bboxes_224", None)
                            bb_224 = v_meta[j].get("bbox_224", None)
                        if not bb_list or (isinstance(bb_list, (list, tuple)) and len(bb_list) == 0):
                            bb_list = []
                            if bb_224 is not None and not _is_no_box_sentinel(bb_224):
                                bb_list = [bb_224]
                        if include_global:
                            bb_list = list(bb_list) + [None]
                        crops = []
                        if isinstance(bb_list, (list, tuple)) and len(bb_list) > 0:
                            for bb in bb_list:
                                if bb is None or (isinstance(bb, (list, tuple)) and len(bb) >= 4 and all(v == -1 for v in bb[:4])):
                                    continue
                                crops.append(_crop_and_resize_tensor(img_j, bb, out_hw=(224, 224)))
                        if not crops:
                            crops = [_crop_and_resize_tensor(img_j, None, out_hw=(224, 224))]
                        counts.append(len(crops))
                        all_crops.extend(crops)
                    all_crops_t = torch.stack(all_crops, dim=0).to(device)
                    all_logits = model(all_crops_t)
                    v_logits_list = []
                    idx0 = 0
                    for c in counts:
                        logits_k = all_logits[idx0:idx0+c]
                        if roi_agg == "max":
                            logits = logits_k.max(dim=0, keepdim=False).values
                        else:
                            logits = logits_k.mean(dim=0, keepdim=False)
                        v_logits_list.append(logits)
                        idx0 += c
                    v_outputs = torch.stack(v_logits_list, dim=0)
                    v_preds = torch.argmax(v_outputs, dim=1).cpu().numpy()
                    all_v_preds.extend(v_preds)
                    all_v_labels.extend(v_labels.detach().cpu().numpy())
            from sklearn.metrics import accuracy_score as _acc
            val_acc = _acc(all_v_labels, all_v_preds)
            val_acc_history.append(val_acc)
            msg += f", ValAcc = {val_acc:.4f}"
            if save_best_path is not None and val_acc > best_val_acc:
                torch.save(model.state_dict(), save_best_path)
                best_val_acc = val_acc
                logger.info(f"[checkpoint] Saved best weights (val_acc={val_acc:.4f})")
            model.train()

        logger.info(msg)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(loss_history, label='Loss'); plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(acc_history, label='Accuracy');
    if val_acc_history:
        plt.plot(val_acc_history, label='Val Accuracy')
    plt.title("Training Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig("training_curves.png"); plt.show()

    # no persistent CAM hooks to release

    return model
