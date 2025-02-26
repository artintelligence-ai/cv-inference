from allenai_common import Registrable

from settings.base import ExtraFieldsNotAllowedBaseSettings


class BasePredictionStrategySettings(ExtraFieldsNotAllowedBaseSettings): ...


class BasePredictionStrategySettingsRegistry(Registrable): ...


@BasePredictionStrategySettingsRegistry.register("RotationPredictionStrategy")
class RotationPredictionStrategySettings(BasePredictionStrategySettings): ...
