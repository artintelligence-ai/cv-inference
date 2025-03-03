from ultralytics import YOLO

from service.models.base import BaseModel
from settings.models import YoloDetModelSettings


@BaseModel.register("YoloDet")
class YoloDet(BaseModel):
    def __init__(self, settings: YoloDetModelSettings):
        self.model = YOLO(settings.state_dict)

    def forward(self, x, **kwargs):
        out = self.model(x, **kwargs)
        return out
