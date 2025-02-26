import logging
from ultralytics import YOLO

from service.models.base import BaseModel
from settings.models import YoloDetModelSettings

logger = logging.getLogger(__name__)


@BaseModel.register("YoloDet")
class YoloDet(BaseModel):
    def __init__(self, settings: YoloDetModelSettings):
        logger.info("🔄 Initializing YOLO detection model")
        self.model = YOLO(settings.state_dict)
        logger.info("✅ YOLO detection model initialized")

    def forward(self, x, **kwargs):
        logger.debug("🔄 Running YOLO detection inference")
        out = self.model(x, **kwargs)
        return out
