import base64
import io
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image


def get_pred_labels_from_logits(logits):
    probabilities = torch.softmax(logits.float(), dim=1)
    pred_labels = torch.argmax(probabilities, dim=1)
    return probabilities, pred_labels


def map_predictions(probabilities, pred_labels, labels_map):
    """Map numerical labels and probabilities to class names."""
    label = labels_map[pred_labels.item()]
    probabilities_dict = {
        labels_map[i]: round(prob.item(), 3)
        for i, prob in enumerate(probabilities.squeeze())
    }
    return {"label": label, "probabilities": probabilities_dict}


def convert_image_to_bytes(
    image: np.ndarray, format: str = "PNG", decode=True
) -> bytes:
    """
    Converts a YOLO plot image (NumPy array) to raw bytes.
    """
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    image_bytes = buffer.getvalue()
    if decode:
        image_bytes = base64.b64encode(image_bytes).decode("utf-8")
    return image_bytes


def filter_and_plot(
    orig_image: np.ndarray,
    boxes_cls: list[int],
    boxes_xyxy: list[np.ndarray],
    excluded_class: str,
    class_names: list[str],
    box_color: Union[
        Tuple[int, int, int], dict[str, Tuple[int, int, int]], DictConfig
    ] = (0, 255, 0),
    box_thickness: int = 2,
    text_color: Tuple[int, int, int] = (0, 255, 0),
    text_font_scale: float = 0.5,
) -> np.ndarray:
    """
    Filters bounding boxes to exclude a specified class and plots the remaining ones on the image.

    Args:
        orig_image (np.ndarray): Original image as a NumPy array.
        boxes_cls (list[int]): List of class indices for each bounding box.
        boxes_xyxy (list[np.ndarray]): List of bounding box coordinates in XYXY format.
        excluded_class (str): The class name to exclude from visualization.
        class_names (list[str]): List of class names.
        box_color (Union[Tuple[int, int, int], dict[str, Tuple[int, int, int]], DictConfig]):
            - Single RGB color for all classes.
            - Or a dictionary of class names mapped to their RGB colors.
        box_thickness (int): Thickness of the bounding box lines.
        text_color (Tuple[int, int, int]): RGB color for class text.
        text_font_scale (float): Font scale for class text.

    Returns:
        np.ndarray: The image with filtered bounding boxes drawn.
    """
    # Convert OmegaConf.DictConfig to a standard dictionary if needed
    if isinstance(box_color, DictConfig):
        box_color = dict(box_color)

    # Copy the original image to draw on
    image = orig_image.copy()

    # Filter boxes excluding the specified class
    filtered_boxes = [
        (box, class_names[int(cls_idx.item())])
        for cls_idx, box in zip(boxes_cls, boxes_xyxy)
        if class_names[int(cls_idx.item())] != excluded_class
    ]

    # Check if box_color is a dict
    is_box_dict_color = isinstance(box_color, dict)
    is_txt_dict_color = isinstance(text_color, dict)

    # Draw the filtered bounding boxes
    for box, class_name in filtered_boxes:
        x1, y1, x2, y2 = map(int, box)

        # Determine the color for this class
        if is_box_dict_color:
            color = tuple(
                box_color.get(class_name, (0, 255, 0))
            )  # Convert list to tuple
        else:
            color = box_color  # Use the single color for all classes

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

        # Draw class name text if font scale > 0
        if text_font_scale > 0:
            class_name = "зц" if class_name == "discount" else "списание"

            cv2.putText(
                image,
                class_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_COMPLEX,  # cv2.FONT_HERSHEY_SIMPLEX,
                text_font_scale,
                color,  # text_color,
                1,
                cv2.LINE_AA,
            )

    return image
