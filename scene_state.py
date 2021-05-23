import json
from enum import Enum
import numpy as np

class SceneEvents(str, Enum):
    OBJECT_BETWEEN_DOORS = 'object_between_doors'
    DOORS_NOT_CLOSED = 'doors_not_closed'

class ObjectTypes(str, Enum):
    HUMAN = 'human'
    WEAR = 'wear'
    LIMB = 'limb'
    OTHER = 'other'

class DoorStates(str, Enum):
    OPEN = 'open'
    CLOSED = 'closed'
    SEMI = 'semi'
    UNKNOWN = 'unknown'

class SceneObject:
    def __init__(self, object_type=ObjectTypes.OTHER,
                 position=np.array([0.0, 0.0, 0.0]),
                 dimensions=np.array([0.0, 0.0, 0.0])):
        self.object_type = object_type
        self.position = position
        self.rotation = np.array([0.0, 0.0, 0.0])
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
        }
        return data_json

    def get_box_vertices(self):
        x_bounds = [
            self.position[0] - self.dimensions[0]/2,
            self.position[0] + self.dimensions[0]/2,
        ]
        y_bounds = [
            self.position[1] - self.dimensions[1]/2,
            self.position[1] + self.dimensions[1]/2,
        ]
        z_bounds = [
            self.position[2] - self.dimensions[2]/2,
            self.position[2] + self.dimensions[2]/2,
        ]
        vertices = []
        for x in x_bounds:
            for y in y_bounds:
                for z in z_bounds:
                    vertex = np.array([x,y,z], dtype=np.float)
                    vertices.append(vertex)
        vertices = np.array(vertices)

        return vertices


class SceneState:
    def __init__(self):
        self.events = []
        self.door_open_percent = -1
        self.objects = []
        self.door = DoorStates.UNKNOWN

    def add_event(self, event:SceneEvents):
        if event not in self.events:
            self.events.append(event)

    def add_object(self, object_:SceneObject):
        self.objects.append(object_)

    def set_door_open_percent(self, percent):
        self.door_open_percent = percent

        if percent == 0:
            self.door = DoorStates.CLOSED
        elif percent == 100:
            self.door = DoorStates.OPEN
        elif percent == -1:
            self.door = DoorStates.UNKNOWN
        else:
            self.door = DoorStates.SEMI

        if self.door in [DoorStates.OPEN, DoorStates.SEMI]:
            self.add_event(SceneEvents.DOORS_NOT_CLOSED)

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
            'door_open_percent': self.door_open_percent,
            'door': self.door.value
        }
        return data_json

if __name__ == '__main__':
    state = SceneState()
    state.add_event(SceneEvents.OBJECT_BETWEEN_DOORS)
    state.add_object(SceneObject())

    data_json = state.to_json()
    with open('out.json', 'w') as outf:
        json.dump(data_json, outf)