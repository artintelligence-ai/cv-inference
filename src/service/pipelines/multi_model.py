import asyncio
import logging
from typing import Any, Optional

import torch
from allenai_common import Registrable

from service.models.base import BaseModel
from service.pipelines.base import BasePipeline
from service.prediction_strategies.base import BasePredictionStrategy
from service.processing.base import BaseProcessor
from service.utils import load_state_dict, model2cuda
from settings.models import (
    ClassificationModelSettingsRegistry,
    DetectionModelSettingsRegistry,
)
from settings.pipelines import (
    ModelSettings,
    MultiModelPipelineSettings,
    PredictionSettings,
)
from settings.processing import PostprocessSettingsRegistry, PreprocessSettingsRegistry


@BasePipeline.register("MultiModelPipeline")
class MultiModelPipeline(BasePipeline):
    """
    A pipeline implementation for multiple model inference, supporting parallel model execution.

    This pipeline manages multiple models, their preprocessors, postprocessors, and prediction
    strategies. It handles model initialization and provides a flexible interface for
    making predictions with specific models based on the primary class.
    """

    def __init__(
        self,
        config: MultiModelPipelineSettings,
        device: str,
        local_weights: str,
        logger_name: str | None = None,
    ) -> None:
        """
        Initialize the MultiModelPipeline with configuration settings.

        Args:
            config (MultiModelPipelineSettings): Configuration settings for the pipeline
            device (str): Device to run models on ('cpu' or 'cuda')
            local_weights (str): Path to local model weights
            logger_name (str | None): Custom logger name, defaults to "MultiModelPipeline"
        """
        self._model_settings: dict[str, ModelSettings] = config.model_settings
        self._prediction_settings: dict[str, PredictionSettings] = (
            config.prediction_settings
        )
        self._device: str = device
        self._local_weights: str = local_weights

        # Initialize logger
        self._logger = logging.getLogger(logger_name or "MultiModelPipeline")

        # Initialize component dictionaries as protected attributes
        self._models: dict[str, BaseModel] = {}
        self._preprocessors: dict[str, BaseProcessor] = {}
        self._postprocessors: dict[str, BaseProcessor] = {}
        self._prediction_strategies: dict[str, BasePredictionStrategy] = {}

        self._initialize_components()

    @property
    def models(self) -> dict[str, BaseModel]:
        """Get the dictionary of initialized models."""
        return self._models

    @property
    def preprocessors(self) -> dict[str, BaseProcessor]:
        """Get the dictionary of initialized preprocessors."""
        return self._preprocessors

    @property
    def postprocessors(self) -> dict[str, BaseProcessor]:
        """Get the dictionary of initialized postprocessors."""
        return self._postprocessors

    def _initialize_components(self) -> None:
        """Initialize all pipeline components in the correct order."""
        self._logger.info("🔄 Initializing pipeline components...")
        self._initialize_models()
        self._initialize_processors()
        self._initialize_strategies()
        self._logger.info("✅ All pipeline components initialized successfully")

    def _initialize_models(self) -> None:
        """Initialize models for each primary class based on configuration."""
        self._logger.info("🔄 Initializing models...")
        for primary_class, details in self._model_settings.items():
            try:
                self._logger.info(f"🔄 Loading model for '{primary_class}'")
                model = self._load_and_prepare_model(details)
                self._models[primary_class] = model
                self._logger.info(f"✅ Model for '{primary_class}' loaded successfully")
            except Exception as e:
                self._logger.error(
                    f"❌ Failed to load model for '{primary_class}': {e}"
                )
                raise

    def _initialize_processors(self) -> None:
        """Initialize preprocessors and postprocessors for each primary class."""
        self._logger.info("🔄 Initializing processors...")
        for primary_class, config in self._prediction_settings.items():
            self._initialize_processor_pair(primary_class, config)

    def _initialize_processor_pair(
        self, primary_class: str, config: PredictionSettings
    ) -> None:
        """Initialize both preprocessor and postprocessor for a primary class."""
        try:
            # Initialize preprocessor if specified
            if config.preprocess_name:
                self._logger.info(f"🔄 Initializing preprocessor for '{primary_class}'")
                self._preprocessors[primary_class] = self._create_processor(
                    config.preprocess_name,
                    config.preprocess_args,
                    PreprocessSettingsRegistry,
                )
                self._logger.info(f"✅ Preprocessor initialized for '{primary_class}'")

            # Initialize postprocessor if specified
            if config.postprocess_name:
                self._logger.info(
                    f"🔄 Initializing postprocessor for '{primary_class}'"
                )
                self._postprocessors[primary_class] = self._create_processor(
                    config.postprocess_name,
                    config.postprocess_args,
                    PostprocessSettingsRegistry,
                )
                self._logger.info(f"✅ Postprocessor initialized for '{primary_class}'")

        except Exception as e:
            self._logger.error(
                f"❌ Failed to initialize processors for '{primary_class}': {e}"
            )
            raise

    def _create_processor(self, name: str, args: dict, registry: Any) -> BaseProcessor:
        """Helper method to create a processor instance."""
        return BaseProcessor.by_name(name)(registry.by_name(name)(**args))

    def _initialize_strategies(self) -> None:
        """Initialize prediction strategies for each primary class if defined."""
        self._logger.info("🔄 Initializing strategies...")
        for primary_class, config in self._prediction_settings.items():
            self._initialize_class_strategy(primary_class, config)

    def _initialize_class_strategy(
        self, primary_class: str, config: PredictionSettings
    ) -> None:
        """
        Initialize prediction strategy for a specific primary class.

        Args:
            primary_class: Name of the primary class
            config: Configuration for the strategy
        """
        strategy_name = config.prediction_strategy
        if not strategy_name:
            self._logger.warning(
                f"⚠️ No prediction strategy specified for '{primary_class}'"
            )
            self._prediction_strategies[primary_class] = None
            return

        self._logger.info(f"🔄 Initializing prediction strategy for '{primary_class}'")
        try:
            self._prediction_strategies[primary_class] = BasePredictionStrategy.by_name(
                strategy_name
            )()
            self._logger.info(f"✅ Strategy initialized for '{primary_class}'")
        except Exception as e:
            self._logger.error(
                f"❌ Failed to initialize strategy for '{primary_class}': {e}"
            )
            raise

    def _load_and_prepare_model(self, model_details: ModelSettings):
        """
        Loads and prepares a single model based on the provided model details.

        Args:
            model_details: Configuration details for the model.

        Returns:
            Instantiated model.
        """
        try:
            self._logger.info(f"🔄 Loading model weights: {model_details.model_name}")
            state_dict = load_state_dict(
                bucket_name=model_details.bucket_name,
                path2weights=model_details.model_path,
                download_path=self._local_weights,
                model_name=model_details.model_name,
                torch_ckpt=model_details.torch_ckpt,
                logger=self._logger,
            )

            model = self._create_model_from_task(model_details, state_dict)

            if model_details.model2cuda and self._device != "cpu":
                self._logger.info("🔄 Moving model to CUDA device")
                model = model2cuda(device=self._device, model=model)

            self._logger.info("✅ Model weights loaded successfully")
            return model
        except Exception as e:
            self._logger.error(f"❌ Error loading and preparing the model: {e}")
            raise ValueError("Model loading and preparation failed.") from e

    def _create_model_from_task(self, model_details: ModelSettings, state_dict: Any):
        """
        Factory method to create a model based on the task type specified in model_details.

        Args:
            model_details: Configuration details including task and model class.
            state_dict: State dictionary with model weights.

        Returns:
            Instantiated model.

        Raises:
            ValueError: If the task type is unsupported.
        """
        registry = self._get_registry_for_task(model_details.task)
        model_class = BaseModel.by_name(model_details.model_class)
        config_class = registry.by_name(model_details.model_class)

        self._logger.info(
            f"🔄 Creating {model_details.task} model: {model_details.model_class}"
        )
        return model_class(
            config_class(state_dict=state_dict, **model_details.model_args)
        )

    @staticmethod
    def _get_registry_for_task(task: str) -> type[Registrable]:
        """
        Get the appropriate registry for the given task type.

        Args:
            task: Task type ('classify' or 'detect')

        Returns:
            Registry class for the task

        Raises:
            ValueError: If task type is unsupported
        """
        if task == "classify":
            return ClassificationModelSettingsRegistry
        elif task == "detect":
            return DetectionModelSettingsRegistry
        raise ValueError(f"Unsupported task type: {task}")

    async def _make_prediction(
        self,
        image: Any,
        model: BaseModel,
        preprocessor: Optional[BaseProcessor],
        postprocessor: Optional[BaseProcessor],
        prediction_args: dict,
        **kwargs,
    ) -> dict:
        """
        Make a synchronous prediction in an asynchronous context.

        Args:
            image: Input image
            model: Model to use for prediction
            preprocessor: Optional preprocessor to apply
            postprocessor: Optional postprocessor to apply
            prediction_args: Additional prediction arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed prediction result
        """

        def sync_prediction():
            try:
                self._logger.debug("🔄 Running preprocessing")
                processed_image = (
                    preprocessor(image, **kwargs) if preprocessor else image
                )

                self._logger.debug("🔄 Running model inference")
                with torch.no_grad():
                    out = model.forward(processed_image, **prediction_args)

                if postprocessor:
                    self._logger.debug("🔄 Running post-processing")
                    return postprocessor(out)
                return out
            except Exception as e:
                self._logger.error(f"❌ Base prediction error: {e}")
                raise

        return await asyncio.to_thread(sync_prediction)

    async def predict(self, primary_class: str, image: Any, **kwargs) -> dict:
        """
        Make a prediction for a specific primary class.

        Args:
            primary_class: Name of the primary class to use for prediction
            image: Input image
            **kwargs: Additional keyword arguments

        Returns:
            Prediction result dictionary

        Raises:
            ValueError: If no model is configured for the primary class
        """
        model = self._models.get(primary_class)
        if not model:
            self._logger.error(
                f"❌ No model configured for primary class: '{primary_class}'"
            )
            raise ValueError(
                f"No model configured for primary class: '{primary_class}'"
            )

        preprocessor = self._preprocessors.get(primary_class)
        postprocessor = self._postprocessors.get(primary_class)
        prediction_args = self._prediction_settings[primary_class].prediction_args

        try:
            if self._prediction_strategies.get(primary_class):
                self._logger.info("🔄 Using advanced prediction strategy")
                result = await self._prediction_strategies[primary_class].predict(
                    image, model, preprocessor, postprocessor, prediction_args, **kwargs
                )
            else:
                self._logger.info("🔄 Using basic prediction")
                result = await self._make_prediction(
                    image, model, preprocessor, postprocessor, prediction_args, **kwargs
                )

            self._logger.info("✅ Prediction completed successfully")
            return result
        except Exception as e:
            self._logger.error(f"❌ Prediction error for '{primary_class}': {e}")
            raise
