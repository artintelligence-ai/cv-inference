from typing import Optional, Union

from pydantic import BaseModel


class DetectionObject(BaseModel):
    conf: float
    box: list[float]


class DetectionClass(BaseModel):
    count: int
    objects: list[DetectionObject]
    mean_conf: float = 0.0


class ClassificationOutput(BaseModel):
    label: str
    probability: Union[float, list[float]]
    top5: dict[str, float]
    image: Optional[bytes] = None


class DetectionOutput(BaseModel):
    detected_classes: list[str]
    root: dict[str, DetectionClass]
    image: Optional[bytes] = None
