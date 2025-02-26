import torch.nn as nn
import torchvision.models as models
import torchvision.models.efficientnet as weights

from service.models.base import BaseModel
from settings.models import EfficientNetB4ModelSettings, EfficientNetB6ModelSettings


@BaseModel.register("EfficientNetB6")
class EfficientNetB6(BaseModel, nn.Module):
    def __init__(self, settings: EfficientNetB6ModelSettings):
        super().__init__()
        self.backbone = models.efficientnet_b6(
            weights=weights.EfficientNet_B6_Weights.IMAGENET1K_V1
        )
        self.backbone.classifier[1] = nn.Linear(
            in_features=2304, out_features=settings.num_classes
        )

        self.load_state_dict(settings.state_dict)
        self.eval()

    def forward(self, x):
        out = self.backbone(x)
        return out


@BaseModel.register("EfficientNetB4")
class EfficientNetB4(BaseModel, nn.Module):
    def __init__(self, settings: EfficientNetB4ModelSettings):
        super().__init__()
        self.backbone = models.efficientnet_b4(
            weights=weights.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        self.backbone.classifier[1] = nn.Linear(
            in_features=1792, out_features=settings.num_classes
        )

        self.load_state_dict(settings.state_dict)
        self.eval()

    def forward(self, x):
        out = self.backbone(x)
        return out
