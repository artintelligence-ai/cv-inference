import asyncio
import logging
from typing import Any

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
    PredictionSettings,
    SingleModelPipelineSettings,
)
from settings.processing import PostprocessSettingsRegistry, PreprocessSettingsRegistry


@BasePipeline.register("SingleModelPipeline")
class SingleModelPipeline(BasePipeline):
    """
    A pipeline implementation for single model inference, supporting both classification and detection tasks.

    This pipeline handles model initialization, prediction processing, and optional post-processing steps.
    It can work with different model architectures and supports both basic and strategy-based prediction approaches.
    """

    def __init__(
        self,
        config: SingleModelPipelineSettings,
        device: str,
        local_weights: str,
        logger_name: str | None = None,
    ) -> None:
        """
        Initialize the SingleModelPipeline.

        Args:
            config (SingleModelPipelineSettings): Pipeline configuration settings
            device (str): Device to run the model on ('cpu' or 'cuda')
            local_weights (str): Path to local model weights
            logger_name (str | None): Custom logger name, defaults to "SingleModelPipeline"
        """
        self._model_settings: ModelSettings = config.model_settings
        self._prediction_settings: PredictionSettings = config.prediction_settings
        self._device: str = device
        self._local_weights: str = local_weights

        # Initialize logger
        self._logger = logging.getLogger(logger_name or "SingleModelPipeline")

        # Initialize components as protected attributes
        self._model: BaseModel | None = None
        self._preprocessor: BaseProcessor | None = None
        self._postprocessor: BaseProcessor | None = None
        self._prediction_strategy: BasePredictionStrategy | None = None

        self._initialize_components()

    @property
    def model(self) -> BaseModel | None:
        """Get the initialized model instance."""
        return self._model

    @property
    def postprocessor(self) -> BaseProcessor | None:
        """Get the initialized postprocessor instance."""
        return self._postprocessor

    def _initialize_components(self) -> None:
        """Initialize all pipeline components in the correct order."""
        self._logger.info("🔄 Initializing pipeline components...")
        self._initialize_model()
        self._initialize_processors()
        self._initialize_strategies()
        self._logger.info("✅ All components initialized successfully")

    def _initialize_model(self) -> None:
        """Initialize the model based on configuration settings."""
        self._logger.info("🔄 Initializing model...")
        try:
            self._model = self._load_and_prepare_model(self._model_settings)
            self._logger.info("✅ Model initialized successfully")
        except Exception as e:
            self._logger.error(f"❌ Failed to load the model: {e}")
            raise

    def _initialize_processors(self) -> None:
        """Initialize the preprocessor and postprocessor if specified in settings."""
        # Initialize preprocessor
        preprocess_name = self._prediction_settings.preprocess_name
        if not preprocess_name:
            self._logger.warning("⚠️ No preprocessor specified in the configuration")
        else:
            self._logger.info(f"🔄 Initializing preprocessor: {preprocess_name}")
            try:
                self._preprocessor = BaseProcessor.by_name(preprocess_name)(
                    PreprocessSettingsRegistry.by_name(preprocess_name)(
                        **self._prediction_settings.preprocess_args
                    )
                )
                self._logger.info("✅ Preprocessor initialized successfully")
            except Exception as e:
                self._logger.error(f"❌ Failed to initialize preprocessor: {e}")
                raise

        # Initialize postprocessor
        postprocess_name = self._prediction_settings.postprocess_name
        if not postprocess_name:
            self._logger.warning("⚠️ No postprocessor specified in the configuration")
            return

        self._logger.info(f"🔄 Initializing postprocessor: {postprocess_name}")
        try:
            self._postprocessor = BaseProcessor.by_name(postprocess_name)(
                PostprocessSettingsRegistry.by_name(postprocess_name)(
                    **self._prediction_settings.postprocess_args
                )
            )
            self._logger.info("✅ Postprocessor initialized successfully")
        except Exception as e:
            self._logger.error(f"❌ Failed to initialize postprocessor: {e}")
            raise

    def _initialize_strategies(self) -> None:
        """Initialize prediction strategy if specified in settings."""
        strategy_name = self._prediction_settings.prediction_strategy
        if not strategy_name:
            self._logger.warning(
                "⚠️ No prediction strategy specified in the configuration"
            )
            return

        self._logger.info(f"🔄 Initializing prediction strategy: {strategy_name}")
        try:
            self._prediction_strategy = None  # Placeholder for strategy registry
            self._logger.info("✅ Prediction strategy initialized successfully")
        except Exception as e:
            self._logger.error(f"❌ Failed to initialize prediction strategy: {e}")
            raise

    def _load_and_prepare_model(self, model_details: ModelSettings):
        """
        Loads and prepares the model based on the provided model details.

        Args:
            model_details: Configuration details for the model.

        Returns:
            Model instance with loaded state dict.
        """
        try:
            self._logger.info(f"🔄 Loading model weights: {model_details.model_name}")
            # Load the model state dictionary
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
            raise

    def _create_model_from_task(
        self, model_details: ModelSettings, state_dict: Any
    ) -> BaseModel:
        """
        Create a model instance based on the specified task type.

        Args:
            model_details: Model configuration details
            state_dict: Model weights state dictionary

        Returns:
            Instantiated model

        Raises:
            ValueError: If task type is unsupported
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
        prediction_args: dict = {},
    ) -> dict:
        """
        Make a synchronous prediction in an asynchronous context.

        Args:
            image: Input image
            prediction_args: Additional prediction arguments

        Returns:
            Processed prediction result
        """

        def sync_prediction():
            try:
                with torch.no_grad():
                    self._logger.debug("🔄 Running model inference")
                    # Apply preprocessing if available
                    if self._preprocessor:
                        self._logger.debug("🔄 Running pre-processing")
                        processed_image = self._preprocessor(image)
                    else:
                        processed_image = image

                    # Run model inference
                    out = self._model.forward(processed_image, **prediction_args)

                    if self._postprocessor:
                        self._logger.debug("🔄 Running post-processing")
                        return self._postprocessor(out)
                    return out
            except Exception as e:
                self._logger.error(f"❌ Base prediction error: {e}")
                raise

        return await asyncio.to_thread(sync_prediction)

    async def predict(self, image: Any, **kwargs) -> dict:
        """
        Make a prediction using either basic or strategy-based approach.

        Args:
            image: Input image

        Returns:
            Prediction result dictionary
        """
        prediction_args = self._prediction_settings.prediction_args

        try:
            if self._prediction_strategy:
                self._logger.info("🔄 Using advanced prediction strategy")
                return await self._prediction_strategy.predict(
                    image,
                    self._model,
                    self._preprocessor,
                    self._postprocessor,
                    prediction_args,
                    **kwargs,
                )

            self._logger.info("🔄 Using basic prediction")
            result = await self._make_prediction(
                image,
                prediction_args,
                **kwargs,
            )
            self._logger.info("✅ Prediction completed successfully")
            return result
        except Exception as e:
            self._logger.error(f"❌ Prediction error: {e}")
            raise
