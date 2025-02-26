from allenai_common import Registrable

from settings.base import ExtraFieldsNotAllowedBaseSettings


class BaseProcessingSettings(ExtraFieldsNotAllowedBaseSettings): ...


class PostprocessSettingsRegistry(Registrable): ...


class PreprocessSettingsRegistry(Registrable): ...


@PreprocessSettingsRegistry.register("ImagePreprocessor")
class ImagePreprocessorSettings(BaseProcessingSettings):
    apply_transform: bool
    normalization: bool
    img_size: int
    detection_mask: bool


@PostprocessSettingsRegistry.register("YoloClsPostProcessor")
class YoloClsPostProcessorSettings(BaseProcessingSettings):
    top5: bool


@PostprocessSettingsRegistry.register("YoloPostProcessor")
class YoloPostProcessorSettings(BaseProcessingSettings):
    boxtype: str
    return_out: bool


@PostprocessSettingsRegistry.register("YoloDet2ClsPostProcessor")
class YoloDet2ClsPostProcessorSettings(BaseProcessingSettings):
    boxtype: str
    det2cls_percentage_thr: dict
    visualization_params: dict


@PostprocessSettingsRegistry.register("CustomClassificationPostProcessor")
class CustomClassificationPostProcessorSettings(BaseProcessingSettings):
    labels_map: dict
    include_all_probabilities: bool
