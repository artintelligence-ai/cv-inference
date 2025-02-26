from abc import ABC, abstractmethod
from typing import Any

from allenai_common import Registrable


class BasePredictionStrategy(ABC, Registrable):
    """
    Abstract base class for prediction strategies.
    """

    @abstractmethod
    def predict(
        self,
        image: Any,
        model: Any,
        preprocessor: Any,
        postprocessor: Any,
        prediction_args: dict,
    ) -> dict:
        """
        Perform prediction using the given model and preprocess/postprocess pipeline.

        Args:
            image (Any): Input image for prediction.
            model (Any): The model to use for prediction.
            preprocessor (Any): Preprocessing function or object.
            postprocessor (Any): Postprocessing function or object.
            prediction_args (dict): Additional arguments for prediction.

        Returns:
            Dict: Processed prediction result.
        """
        pass
