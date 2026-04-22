"""
SVGC (Stochastic Visible-to-Grayscale Conversion) Augmentation for UAV Small Object Detection.

SVGC is a stochastic augmentation technique for cross-modality generalization.
It equalizes the histogram of the Value channel (from HSV) to simulate
grayscale-like or infrared-like imagery, which helps the model generalize
across different lighting conditions or sensor modalities.

Source paper: RGB-IR Object Detection (Gautam et al., IIT Goa, ETAAV 2025)
"""

import cv2
import numpy as np
import albumentations as A
import random


def apply_svgc(image_rgb, p=0.5):
    """
    Applies SVGC augmentation to a single RGB image.
    
    Args:
        image_rgb: HxWx3 uint8 RGB numpy array.
        p: Probability of applying the transformation.
        
    Returns:
        Augmented or original HxWx3 uint8 RGB numpy array.
    """
    if random.random() > p:
        return image_rgb
        
            
    # Convert RGB -> HSV
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Extract V channel (index 2)
    V = hsv[:, :, 2]
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    V_eq = clahe.apply(V.astype(np.uint8))
    
    # Stack [V_eq, V_eq, V_eq] to form 3-channel image
    return np.stack([V_eq, V_eq, V_eq], axis=-1)


class SVGC(A.ImageOnlyTransform):
    """
    Albumentations-compatible SVGC transform.
    """
    def __init__(self, p=0.5):
        super().__init__(p=p)
        
    def apply(self, img, **params):
        # Albumentations handles probability externally, so always convert inside apply()
        return apply_svgc(img, p=1.0)
        
    def get_transform_init_args_names(self):
        return ("p",)


if __name__ == '__main__':
    # Quick test on a random 640x640x3 array
    img_test = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    out_test = apply_svgc(img_test, p=1.0)
    assert out_test.shape == img_test.shape, "Output shape mismatch!"
    print("SVGC module test passed!")
