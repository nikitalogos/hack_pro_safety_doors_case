import json
from enum import Enum
import numpy as np

class SceneEvents(str, Enum):
    OBJECT_BETWEEN_DOORS = 'object_between_doors'
    OBJECT_BEYOND_SAFE_LINE = 'object_beyond_safe_line'
    DOORS_NOT_CLOSED = 'doors_not_closed'

class ObjectTypes(str, Enum):
    HUMAN = 'human'
    WEAR = 'wear'
    LIMB = 'limb'
    OTHER = 'other'

class SceneObject:
    def __init__(self, object_type=ObjectTypes.OTHER,
                 door='unknown',
                 position=np.array([0.0, 0.0, 0.0]),
                 rotation=np.array([0.0, 0.0, 0.0]),
                 dimensions=np.array([0.0, 0.0, 0.0])):
        self.object_type = object_type
        self.door = door
        self.position = position
        self.rotation = rotation
        self.dimensions = dimensions

    def to_json(self):
        data_json = {
            'object': self.object_type.value,
            'geometry': {
                'position': {
                    'x': self.position[0],
                    'y': self.position[1],
                    'z': self.position[2],
                },
                'rotation': {
                    'x': self.rotation[0],
                    'y': self.rotation[1],
                    'z': self.rotation[2],
                },
                'dimensions': {
                    'x': self.dimensions[0],
                    'y': self.dimensions[1],
                    'z': self.dimensions[2],
                },
            },
            'door': self.door
        }
        return data_json


class SceneState:
    def __init__(self):
        self.events = []
        self.door_open_percent = -1
        self.objects = []

    def add_event(self, event:SceneEvents):
        if event not in self.events:
            self.events.append(event)

    def add_object(self, object_:SceneObject):
        self.objects.append(object_)

    def is_can_move(self):
        return len(self.events) == 0

    def to_json(self):
        figures = []
        for obj in self.objects:
            figure = obj.to_json()
            figures.append(figure)

        data_json = {
            'figures': figures,
            'events': [event.value for event in self.events],
            'is_can_move': self.is_can_move(),
            'door_open_percent': self.door_open_percent
        }
        return data_json

if __name__ == '__main__':
    state = SceneState()
    state.add_event(SceneEvents.OBJECT_BETWEEN_DOORS)
    state.add_object(SceneObject())

    data_json = state.to_json()
    with open('out.json', 'w') as outf:
        json.dump(data_json, outf)