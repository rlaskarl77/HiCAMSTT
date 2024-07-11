from dataclasses import dataclass
from typing import List

@dataclass
class ObjectType:
    type: int
    id: int
    action: int
    value: int
    posx: float
    posy: float
    posz: float
    sizex: float
    sizey: float
    sizez: float
    execution: int

@dataclass
class Camera:
    camera_id: str
    object_type: List[ObjectType]

@dataclass
class Data:
    time: int
    camera: List[Camera]