from typing import Any

from allenai_common import Registrable

from settings.base import ExtraFieldsNotAllowedBaseSettings


class BaseModelSettings(ExtraFieldsNotAllowedBaseSettings):
    state_dict: Any


class ClassificationModelSettingsRegistry(Registrable): ...


class DetectionModelSettingsRegistry(Registrable): ...


@ClassificationModelSettingsRegistry.register("EfficientNetB6")
class EfficientNetB6ModelSettings(BaseModelSettings):
    num_classes: int


@ClassificationModelSettingsRegistry.register("EfficientNetB4")
class EfficientNetB4ModelSettings(BaseModelSettings):
    num_classes: int


@ClassificationModelSettingsRegistry.register("YoloCls")
class YoloClsModelSettings(BaseModelSettings): ...


@DetectionModelSettingsRegistry.register("YoloDet")
class YoloDetModelSettings(BaseModelSettings): ...
