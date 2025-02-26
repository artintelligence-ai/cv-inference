import logging

from ultralytics import YOLO

from service.models.base import BaseModel
from settings.models import YoloClsModelSettings

logger = logging.getLogger(__name__)


@BaseModel.register("YoloCls")
class YoloCls(BaseModel):
    def __init__(self, settings: YoloClsModelSettings):
        logger.info("🔄 Initializing YOLO classification model")
        self.model = YOLO(settings.state_dict)
        logger.info("✅ YOLO classification model initialized")

    def forward(self, x, **kwargs):
        logger.debug("🔄 Running YOLO classification inference")
        out = self.model(x, **kwargs)
        return out
