"""
Attention heatmap generation for YOLOv8 models.

This helper captures activations from a chosen backbone layer and projects
them into a stable heatmap overlay. It is more reliable in Colab than using
third-party Grad-CAM wrappers directly against YOLOv8 internals.
"""

import cv2
import numpy as np
import torch


def _unwrap_tensor(output):
    """Return the first tensor from nested tuples/lists produced by YOLO."""
    while isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Target layer returned an empty tuple/list.")
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"Expected a tensor-like activation, got {type(output).__name__}."
        )
    return output


def generate_attention_heatmap(model, image_path, target_layer, save_path):
    """
    Generate a stable attention heatmap for a single image.

    Args:
        model: Ultralytics YOLO model with a loaded checkpoint.
        image_path: Path to the input image file.
        target_layer: Backbone layer to hook for feature activations.
        save_path: Output path for the heatmap PNG file.

    Returns:
        numpy.ndarray: BGR image with heatmap overlay.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_resized = cv2.resize(img_bgr, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb /= 255.0

    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
    device = next(model.model.parameters()).device
    img_tensor = img_tensor.to(device)

    activation = {}

    def hook_fn(_, __, output):
        activation["feat"] = _unwrap_tensor(output).detach()

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        model.model.eval()
        with torch.no_grad():
            _ = model.model(img_tensor)
    finally:
        handle.remove()

    feat = activation.get("feat")
    if feat is None:
        raise RuntimeError("Failed to capture activations from the target layer.")

    heatmap = feat[0].mean(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap = cv2.resize(heatmap, (640, 640))
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized, 0.55, heatmap_color, 0.45, 0)

    cv2.imwrite(save_path, overlay)
    return overlay
