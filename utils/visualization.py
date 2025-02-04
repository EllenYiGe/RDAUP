import torch
import torch.nn.functional as F
import numpy as np
import cv2

def grad_cam_example():
    """
    Placeholder Grad-CAM example function, can add visualization layers/hooks
    based on the actual network.
    """
    pass

def plot_attention_map(img_np, attention_map):
    """
    Overlay the attention_map onto the original image:
      - img_np: (H, W, 3) numpy image data
      - attention_map: (H, W) range [0,1], or [0,255]
    Returns: The fused heatmap
    """
    if attention_map.max() <= 1.0:
        attention_map = attention_map * 255.0
    attention_map = np.uint8(attention_map)
    heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
    out = 0.5 * heatmap + 0.5 * img_np
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
