"""
Grad-CAM heatmap generation for YOLOv8 models.

This module now wraps heatmap_utils.py to provide a stable,
activation-based attention heatmap generation without relying on
third-party Grad-CAM wrappers.
"""

from .heatmap_utils import generate_attention_heatmap

def generate_gradcam(model, image_path, target_layer, save_path):
    """
    Generate an attention heatmap for a single image, wrapping the stable
    heatmap generation for backward compatibility with older notebook flows.

    Args:
        model: Ultralytics YOLO model with a loaded checkpoint.
        image_path: Path to the input image file.
        target_layer: PyTorch module to compute gradients with respect to.
        save_path: Output path for the heatmap PNG file.

    Returns:
        numpy.ndarray: The visualization image (RGB) with heatmap overlay.
    """
    return generate_attention_heatmap(model, image_path, target_layer, save_path)
