from PIL import ImageDraw
import numpy as np
import cv2


def draw_boxes_on_image(img, boxes_xyxy, color=(0, 0, 0)):
    d = ImageDraw.Draw(img)
    for (x1, y1, x2, y2) in boxes_xyxy:
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return img


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
    from PIL import Image
    return Image.fromarray(overlay)

