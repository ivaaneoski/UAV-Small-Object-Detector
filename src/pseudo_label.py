"""
Pseudo-labeling pipeline for semi-supervised UAV object detection.

Uses a trained YOLOv8 model to auto-generate labels for unlabeled aerial
images by keeping only high-confidence predictions. This reduces manual
annotation burden for drone datasets while expanding the effective training
set size.
"""

import os

from ultralytics import YOLO


def generate_pseudo_labels(model_path, unlabeled_dir, output_dir, conf=0.5):
    """
    Run YOLOv8 inference on unlabeled images and save YOLO-format labels
    for predictions above the confidence threshold.

    Low-confidence predictions are rejected, implementing a conservative
    pseudo-labeling strategy that prioritizes label quality over quantity.

    Args:
        model_path: Path to the trained YOLOv8 .pt weights file.
        unlabeled_dir: Directory containing unlabeled image files (.jpg/.png).
        output_dir: Directory where generated .txt label files will be saved.
        conf: Minimum confidence threshold for accepting a detection (default 0.5).

    Returns:
        tuple: (accepted_count, rejected_count) — number of images for which
            pseudo-labels were generated vs rejected due to low confidence.

    Output format:
        Each .txt file contains one detection per line:
            <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

        Values are normalized to [0, 1] range, compatible with YOLO training.
    """
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    accepted, rejected = 0, 0

    for img_name in os.listdir(unlabeled_dir):
        img_path = os.path.join(unlabeled_dir, img_name)
        results = model(img_path, conf=conf, verbose=False)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(output_dir, label_name)

        boxes = results[0].boxes
        if len(boxes) == 0:
            rejected += 1
            continue

        with open(label_path, "w") as f:
            for box in boxes:
                cls = int(box.cls[0])
                xywhn = box.xywhn[0].tolist()
                f.write(f"{cls} {' '.join(map(str, xywhn))}\n")
        accepted += 1

    print(f"Accepted: {accepted} | Rejected (low conf): {rejected}")
    print(
        f"Pseudo-label acceptance rate: {accepted/(accepted+rejected)*100:.1f}%"
    )
    return accepted, rejected
