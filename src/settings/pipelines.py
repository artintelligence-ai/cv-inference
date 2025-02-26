from typing import Literal, Optional

from allenai_common import Registrable

from settings.base import ExtraFieldsNotAllowedBaseSettings


class BasePipelineSettings(ExtraFieldsNotAllowedBaseSettings):
    enable: bool
    pipeline_class: str
    logger_name: str


class PredictionSettings(ExtraFieldsNotAllowedBaseSettings):
    preprocess_name: Optional[str] = None
    preprocess_args: dict = {}
    prediction_strategy: Optional[str] = None
    prediction_args: dict = {}
    postprocess_name: Optional[str] = None
    postprocess_args: dict = {}


class ModelSettings(ExtraFieldsNotAllowedBaseSettings):
    task: Literal["detect", "classify"]
    model_class: str
    model_name: str
    bucket_name: str
    model_path: str
    torch_ckpt: bool
    model2cuda: bool
    model_args: dict = {}


class BasePipelineSettingsRegistry(Registrable): ...


@BasePipelineSettingsRegistry.register("SingleModelPipeline")
class SingleModelPipelineSettings(BasePipelineSettings):
    prediction_settings: PredictionSettings
    model_settings: ModelSettings


@BasePipelineSettingsRegistry.register("MultiModelPipeline")
class MultiModelPipelineSettings(BasePipelineSettings):
    prediction_settings: dict[str, PredictionSettings]
    model_settings: dict[str, ModelSettings]
