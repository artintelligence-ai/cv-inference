from ultralytics import YOLO

from service.models.base import BaseModel
from settings.models import YoloClsModelSettings


@BaseModel.register("YoloCls")
class YoloCls(BaseModel):
    def __init__(self, settings: YoloClsModelSettings):
        self.model = YOLO(settings.state_dict)

    def forward(self, x, **kwargs):
        out = self.model(x, **kwargs)
        return out
