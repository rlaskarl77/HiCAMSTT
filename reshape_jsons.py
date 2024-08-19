from dataclasses import dataclass, asdict
from typing import List
import json
import os
import glob
from tqdm import tqdm

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
    objects: List[ObjectType]

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
        data_dict = data_dict[0] if isinstance(data_dict, list) else data_dict
        for camera in data_dict['camera']:
            camera['objects'] = [ObjectType(**obj) for obj in camera['objects']]
        data_dict['camera'] = [Camera(**camera) for camera in data_dict['camera']]
        return Data(**data_dict)

# list all json files in the directory
def list_json_files(directory: str) -> List[str]:
    return glob.glob(os.path.join(directory, '*.json'))

# read json file
def read_json_file(filename: str) -> str:
    with open(filename, 'r') as file:
        return file.read()

from typing import Union

def change_camera_id(data: Data, new_id: Union[str, int]) -> Data:
    for camera in data.camera:
        camera.camera_id = new_id
    return data

def make_as_list_of_dicts(data: Data) -> List[dict]:
    return [asdict(data)]

def save_as_list_of_dicts(data: Data, filename: str) -> None:
    with open(filename, 'w') as file:
        json.dump(make_as_list_of_dicts(data), file, indent=4)

def main(path: str = None):
    
    json_list = list_json_files(path)
    
    for json_file in tqdm(json_list):
        data = Data.from_json(read_json_file(json_file))
        data = change_camera_id(data, -1)
        save_as_list_of_dicts(data, json_file)

if __name__ == '__main__':
    
    path = '/131_data/namgi/logs/HDC/version_143'
    main(path)
    path = '/131_data/namgi/logs/HDC/version_144'
    main(path)