from typing import Any, Optional, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from service.constants.image_constants import MEAN, STD
from service.processing.base import BaseProcessor
from settings.processing import ImagePreprocessorSettings


@BaseProcessor.register("ImagePreprocessor")
class ImagePreprocessor(BaseProcessor):
    """
    Preprocessor for image transformations, including resizing, normalization, and optional masking using bounding boxes.
    """

    def __init__(self, settings: ImagePreprocessorSettings):
        """
        Initializes the preprocessor with the desired settings.

        Args:
            settings (ImagePreprocessorSettings): Preprocessing settings.
        """
        self.settings = settings
        self.transform = self._create_transform() if settings.apply_transform else None

    def _create_transform(self) -> transforms.Compose:
        """
        Creates a composed transform pipeline based on settings.

        Returns:
            transforms.Compose: Transformation pipeline.
        """
        steps = []
        if self.settings.img_size:
            steps.append(
                transforms.Resize((self.settings.img_size, self.settings.img_size))
            )
        steps.append(transforms.ToTensor())
        if self.settings.normalization:
            steps.append(transforms.Normalize(MEAN, STD))
        return transforms.Compose(steps)

    @staticmethod
    def _apply_mask(image: Image.Image, bboxes: list[dict]) -> Image.Image:
        """
        Masks the image, blacking out areas outside bounding boxes.

        Args:
            image (Image.Image): Input PIL image.
            bboxes (List[dict]): List of bounding boxes as dictionaries with "box" key.

        Returns:
            Image.Image: Masked image.
        """

        image_np = np.array(image)
        mask = np.zeros_like(image_np)

        for object in bboxes:
            x1, y1, x2, y2 = map(int, object.box)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)

            mask[y1:y2, x1:x2] = image_np[y1:y2, x1:x2]

        return Image.fromarray(mask)

    def __call__(
        self, image: Any, bboxes: Optional[list[dict]] = None
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Preprocesses the image with optional masking and transformations.

        Args:
            image (Any): Input image (PIL or NumPy).
            bboxes (Optional[List[dict]]): Bounding boxes for masking.

        Returns:
            Union[Image.Image, torch.Tensor]: Preprocessed image as a PIL.Image or tensor.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if self.settings.detection_mask and bboxes:
            image = self._apply_mask(image, bboxes)

        if self.transform:
            image = self.transform(image).unsqueeze(0)

        return image
