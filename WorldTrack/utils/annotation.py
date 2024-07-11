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
    time: str
    camera: List[Camera]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)

    def save_to_file(self, filename: str) -> None:
        with open(filename, 'w') as file:
            json.dump(asdict(self), file, indent=4)

    @staticmethod
    def from_json(json_str: str) -> 'Data':
        data_dict = json.loads(json_str)
        for camera in data_dict['camera']:
            camera['object_type'] = [ObjectType(**obj) for obj in camera['object_type']]
        data_dict['camera'] = [Camera(**camera) for camera in data_dict['camera']]
        return Data(**data_dict)