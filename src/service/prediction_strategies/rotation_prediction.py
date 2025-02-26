import asyncio
from typing import Any

import torch

from schema.schema import ClassificationOutput
from service.prediction_strategies.base import BasePredictionStrategy


@BasePredictionStrategy.register("RotationPredictionStrategy")
class RotationPredictionStrategy(BasePredictionStrategy):
    """
    A prediction strategy that evaluates the model on multiple rotations of the input image.
    """

    def __init__(self, rotations=None):
        """
        Initialize the strategy with optional rotation angles.

        Args:
            rotations (list, optional): List of rotation angles. Defaults to [0, 90, 180, 270].
        """
        self.rotations = rotations or [0, 90, 180, 270]

    def _rotate_image(self, image, angle: int) -> Any:
        """
        Rotate the input image by the specified angle.

        Args:
            image (Any): Input image.
            angle (int): Angle to rotate the image.

        Returns:
            Any: Rotated image.
        """
        return image.rotate(angle, expand=True)

    async def predict(
        self,
        image: Any,
        model: Any,
        preprocessor: Any,
        postprocessor: Any,
        prediction_args: dict,
        **kwargs,
    ) -> dict:
        """
        Perform predictions on multiple rotations of the input image and choose the best result.

        Args:
            image (Any): Input image for prediction.
            model (Any): The model to use for prediction.
            preprocessor (Any): Preprocessing function or object.
            postprocessor (Any): Postprocessing function or object.
            prediction_args (dict): Additional arguments for prediction.

        Returns:
            dict: Processed prediction result.
        """

        def sync_prediction_for_rotation(image, rotation):
            """Perform synchronous prediction for a single rotation."""
            try:
                # Rotate the image
                rotated_image = self._rotate_image(image, rotation)

                # Preprocess the image
                if preprocessor:
                    rotated_image = preprocessor(rotated_image, **kwargs)

                # Model inference
                with torch.no_grad():
                    out = model(rotated_image, **prediction_args)

                # Postprocess the prediction
                if postprocessor:
                    prediction = postprocessor(out)
                else:
                    prediction = out

                return prediction
            except Exception as e:
                raise ValueError(
                    f"Error during synchronous prediction for rotation {rotation}: {e}"
                )

        prediction_counts = {}
        prediction_probabilities = {}
        prediction_with_rotations = {}

        try:
            # Run predictions for all rotations asynchronously
            predictions = await asyncio.gather(
                *[
                    asyncio.to_thread(sync_prediction_for_rotation, image, rotation)
                    for rotation in self.rotations
                ]
            )

            # Process all predictions
            for rotation, prediction in zip(self.rotations, predictions):
                label = prediction.label
                probability = prediction.probability

                prediction_counts[label] = prediction_counts.get(label, 0) + 1
                prediction_with_rotations[str(rotation)] = prediction

                if label not in prediction_probabilities:
                    prediction_probabilities[label] = []
                prediction_probabilities[label].append(probability)

            # Choose the final label
            final_label, max_count = max(prediction_counts.items(), key=lambda x: x[1])
            if max_count >= 3:
                chosen_label = final_label
            elif all(count == 2 for count in prediction_counts.values()):
                avg_probabilities = {
                    label: sum(probs) / len(probs)
                    for label, probs in prediction_probabilities.items()
                }
                chosen_label = max(avg_probabilities, key=avg_probabilities.get)
            else:
                chosen_label = final_label

        except Exception as e:
            raise ValueError(f"Error during advanced prediction: {e}")

        return ClassificationOutput(
            label=chosen_label,
            probability=prediction_probabilities[chosen_label],
            top5={},
        )
