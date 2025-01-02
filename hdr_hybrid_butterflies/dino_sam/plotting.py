"""Adopted from:

https://github.com/NielsRogge/Transformers-Tutorials/
blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb

See LICENSE in this directory for more details.
"""
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from hdr_hybrid_butterflies.dino_sam.dino_sam import (
    DetectionResult,
)


def annotate(
    image: Union[Image.Image, np.ndarray],
    detection_results: List[DetectionResult],
    width: int = 20,
) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    print(len(detection_results))

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        print(detection.label, detection.score)
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(
            image_cv2,
            (box.xmin, box.ymin),
            (box.xmax, box.ymax),
            color.tolist(),
            thickness=width,
        )
        cv2.putText(
            image_cv2,
            f"{label}: {score:.2f}",
            (box.xmin, box.ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            color.tolist(),
            thickness=width,
        )

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                image_cv2, contours, -1, color.tolist(), thickness=width
            )

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> None:
    annotated_image = annotate(image, detections)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(annotated_image)
    ax.axis("off")
    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")
    return fig, ax
