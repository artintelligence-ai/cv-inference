import abc
import logging
from typing import Any, Dict, Optional

from allenai_common import Registrable


class BasePipeline(abc.ABC, Registrable):
    """
    Abstract base class for model inference pipelines.

    Provides a common interface for both single and multi-model pipelines,
    with abstract methods that must be implemented by subclasses.
    """

    def __init__(self, logger_name: Optional[str] = None):
        """Initialize with optional custom logger name."""
        self._logger = logging.getLogger(logger_name or self.__class__.__name__)

    @abc.abstractmethod
    def _initialize_components(self) -> None:
        """
        Initialize all pipeline components (models, processors, strategies).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _initialize_components")

    @abc.abstractmethod
    async def predict(self, image: Any, **kwargs) -> Dict[str, Any]:
        """
        Make a prediction using the initialized pipeline.

        Args:
            image: Input image to process
            **kwargs: Additional keyword arguments for prediction

        Returns:
            Dictionary containing prediction results

        Raises:
            NotImplementedError: If not implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement predict")
