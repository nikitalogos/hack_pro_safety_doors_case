#!/bin/sh
"exec" "`dirname $0`/venv/bin/python" "$0" "$@"
#ГОЛУБОЕ-ЧЕЛОВЕК
#КРАСНОЕ-КОНЕЧНОСТЬ
#ЗЕЛЕНОЕ-ДРУГОЕ
#ЧЕРНОЕ-ОДЕЖДА

import open3d as o3d
import json
import numpy as np
import os
from argparse import ArgumentParser
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion

def text_3d(text, pos, direction=None, degree=0.0, font='arial.ttf', font_size=16,density=10):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)
    if os.name == 'posix':
        font = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    if text=='Train can not move':
        img = Image.new('RGB', font_dim, color=(255, 0, 0))
    if text == 'Train can move':
        img = Image.new('RGB', font_dim, color=(0, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 800 / density)

    raxis = np.cross([0., 0.5, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.5, -1.0)
   # trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
         #   Quaternion(axis=direction, degrees=degree)).transformation_matrix
    #print(trans)
    trans=[[1.,0.,0.,0.],
            [0.,0.,1.,0.],
            [0.,-1.,0.,0.],
            [0.,0.,0.,1.]]
    #trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    trans=[[0.,0.,-1.,pos[0]],
            [0.,-1.,0.,pos[1]],
            [1.,0.,0.,pos[2]],
            [0.,0.,0.,1.]]
    pcd.transform(trans)
    return pcd

def get_box1(text):
    boxes=[]
    for txt in text["figures"]:
        boxes = []
        # для всех полей с корнем figure
        for txt in text["figures"]:
            init_cord = []
            rot = []
            disp = []
            # если ключь совпадает с кючем для человека
            if txt["object"] == 'limb':
                # записываем координаты начальной точки
                init_cord.append(txt["geometry"]["position"]["x"])
                init_cord.append(txt["geometry"]["position"]["y"])
                init_cord.append(txt["geometry"]["position"]["z"])
                # записываем ротацию
                rot.append(txt["geometry"]["rotation"]["x"])
                rot.append(txt["geometry"]["rotation"]["y"])
                rot.append(txt["geometry"]["rotation"]["z"])
                # записываем переещения
                disp.append(txt["geometry"]["dimensions"]["x"])
                disp.append(txt["geometry"]["dimensions"]["y"])
                disp.append(txt["geometry"]["dimensions"]["z"])
                # координаты углов квадрата
                points = [[init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],

                          [init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          ]
                lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                         [4, 5], [4, 7], [5, 6], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]]

                colors = [[1, 0, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                R = line_set.get_rotation_matrix_from_xyz((rot[0], rot[1], rot[2]))
                line_set = line_set.rotate(R, center=(init_cord[0], init_cord[1], init_cord[2]))
                boxes.append(line_set)
            if txt["object"] == 'wear':
                # записываем координаты начальной точки
                init_cord.append(txt["geometry"]["position"]["x"])
                init_cord.append(txt["geometry"]["position"]["y"])
                init_cord.append(txt["geometry"]["position"]["z"])
                # записываем ротацию
                rot.append(txt["geometry"]["rotation"]["x"])
                rot.append(txt["geometry"]["rotation"]["y"])
                rot.append(txt["geometry"]["rotation"]["z"])
                # записываем переещения
                disp.append(txt["geometry"]["dimensions"]["x"])
                disp.append(txt["geometry"]["dimensions"]["y"])
                disp.append(txt["geometry"]["dimensions"]["z"])
                # координаты углов квадрата
                points = [[init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],

                          [init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          ]
                lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                         [4, 5], [4, 7], [5, 6], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]]

                colors = [[0, 0, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                R = line_set.get_rotation_matrix_from_xyz((rot[0], rot[1], rot[2]))
                line_set = line_set.rotate(R, center=(init_cord[0], init_cord[1], init_cord[2]))
                boxes.append(line_set)
            if txt["object"] == 'human':
                # записываем координаты начальной точки
                init_cord.append(txt["geometry"]["position"]["x"])
                init_cord.append(txt["geometry"]["position"]["y"])
                init_cord.append(txt["geometry"]["position"]["z"])
                # записываем ротацию
                rot.append(txt["geometry"]["rotation"]["x"])
                rot.append(txt["geometry"]["rotation"]["y"])
                rot.append(txt["geometry"]["rotation"]["z"])
                # записываем переещения
                disp.append(txt["geometry"]["dimensions"]["x"])
                disp.append(txt["geometry"]["dimensions"]["y"])
                disp.append(txt["geometry"]["dimensions"]["z"])
                # координаты углов квадрата
                points = [[init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],

                          [init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          ]
                lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                         [4, 5], [4, 7], [5, 6], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]]

                colors = [[0, 0, 1] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                R = line_set.get_rotation_matrix_from_xyz((rot[0], rot[1], rot[2]))
                line_set = line_set.rotate(R, center=(init_cord[0], init_cord[1], init_cord[2]))
                boxes.append(line_set)
            if txt["object"] == 'other':
                # записываем координаты начальной точки
                init_cord.append(txt["geometry"]["position"]["x"])
                init_cord.append(txt["geometry"]["position"]["y"])
                init_cord.append(txt["geometry"]["position"]["z"])
                # записываем ротацию
                rot.append(txt["geometry"]["rotation"]["x"])
                rot.append(txt["geometry"]["rotation"]["y"])
                rot.append(txt["geometry"]["rotation"]["z"])
                # записываем переещения
                disp.append(txt["geometry"]["dimensions"]["x"])
                disp.append(txt["geometry"]["dimensions"]["y"])
                disp.append(txt["geometry"]["dimensions"]["z"])
                # координаты углов квадрата
                points = [[init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],

                          [init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          ]
                lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                         [4, 5], [4, 7], [5, 6], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]]

                colors = [[0, 1, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                R = line_set.get_rotation_matrix_from_xyz((rot[0], rot[1], rot[2]))
                line_set = line_set.rotate(R, center=(init_cord[0], init_cord[1], init_cord[2]))
                boxes.append(line_set)
        return boxes

def get_box(text):
    boxes=[]
    # для всех полей с корнем figure
    for txt in text["figures"]:
            init_cord = []
            rot=[]
            disp = []
         # если ключь совпадает с кючем для человека
            if txt["objectKey"] == limb_id:
                # записываем координаты начальной точки
                init_cord.append(txt["geometry"]["position"]["x"])
                init_cord.append(txt["geometry"]["position"]["y"])
                init_cord.append(txt["geometry"]["position"]["z"])
                # записываем ротацию
                rot.append(txt["geometry"]["rotation"]["x"])
                rot.append(txt["geometry"]["rotation"]["y"])
                rot.append(txt["geometry"]["rotation"]["z"])
                # записываем переещения
                disp.append(txt["geometry"]["dimensions"]["x"])
                disp.append(txt["geometry"]["dimensions"]["y"])
                disp.append(txt["geometry"]["dimensions"]["z"])
                # координаты углов квадрата
                points = [[init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],

                          [init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          ]
                lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                         [4, 5], [4, 7], [5, 6], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]]

                colors = [[1, 0, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                R = line_set.get_rotation_matrix_from_xyz((rot[0], rot[1], rot[2]))
                line_set = line_set.rotate(R, center=(init_cord[0], init_cord[1], init_cord[2]))
                boxes.append(line_set)
            if txt["objectKey"] == wear_id:
                # записываем координаты начальной точки
                init_cord.append(txt["geometry"]["position"]["x"])
                init_cord.append(txt["geometry"]["position"]["y"])
                init_cord.append(txt["geometry"]["position"]["z"])
                # записываем ротацию
                rot.append(txt["geometry"]["rotation"]["x"])
                rot.append(txt["geometry"]["rotation"]["y"])
                rot.append(txt["geometry"]["rotation"]["z"])
                # записываем переещения
                disp.append(txt["geometry"]["dimensions"]["x"])
                disp.append(txt["geometry"]["dimensions"]["y"])
                disp.append(txt["geometry"]["dimensions"]["z"])
                # координаты углов квадрата
                points = [[init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],

                          [init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          ]
                lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                         [4, 5], [4, 7], [5, 6], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]]

                colors = [[0, 0, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                R = line_set.get_rotation_matrix_from_xyz((rot[0], rot[1], rot[2]))
                line_set = line_set.rotate(R, center=(init_cord[0], init_cord[1], init_cord[2]))
                boxes.append(line_set)
            if txt["objectKey"] == human_id:
                # записываем координаты начальной точки
                init_cord.append(txt["geometry"]["position"]["x"])
                init_cord.append(txt["geometry"]["position"]["y"])
                init_cord.append(txt["geometry"]["position"]["z"])
                # записываем ротацию
                rot.append(txt["geometry"]["rotation"]["x"])
                rot.append(txt["geometry"]["rotation"]["y"])
                rot.append(txt["geometry"]["rotation"]["z"])
                # записываем переещения
                disp.append(txt["geometry"]["dimensions"]["x"])
                disp.append(txt["geometry"]["dimensions"]["y"])
                disp.append(txt["geometry"]["dimensions"]["z"])
                # координаты углов квадрата
                points = [[init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],

                          [init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          ]
                lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                         [4, 5], [4, 7], [5, 6], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]]

                colors = [[0, 0, 1] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                R = line_set.get_rotation_matrix_from_xyz((rot[0], rot[1], rot[2]))
                line_set = line_set.rotate(R, center=(init_cord[0], init_cord[1], init_cord[2]))
                boxes.append(line_set)
            if txt["objectKey"] == other_id:
                # записываем координаты начальной точки
                init_cord.append(txt["geometry"]["position"]["x"])
                init_cord.append(txt["geometry"]["position"]["y"])
                init_cord.append(txt["geometry"]["position"]["z"])
                # записываем ротацию
                rot.append(txt["geometry"]["rotation"]["x"])
                rot.append(txt["geometry"]["rotation"]["y"])
                rot.append(txt["geometry"]["rotation"]["z"])
                # записываем переещения
                disp.append(txt["geometry"]["dimensions"]["x"])
                disp.append(txt["geometry"]["dimensions"]["y"])
                disp.append(txt["geometry"]["dimensions"]["z"])
                # координаты углов квадрата
                points = [[init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1], init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1], init_cord[2]+ disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2]+ disp[2]/2],

                          [init_cord[0] - disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1]/2, init_cord[2] - disp[2]/2],
                          [init_cord[0] + disp[0] / 2, init_cord[1]- disp[1]/2, init_cord[2] - disp[2]/2],
                          ]
                lines = [[0, 1], [0, 3], [1, 2], [2, 3],
                         [4, 5], [4, 7], [5, 6], [6, 7],
                         [0, 4], [1, 5], [2, 6], [3, 7]]

                colors = [[0, 1, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                R=line_set.get_rotation_matrix_from_xyz((rot[0],rot[1],rot[2]))
                line_set=line_set.rotate(R,center=(init_cord[0],init_cord[1],init_cord[2]))
                boxes.append(line_set)
    return boxes


def get_o3d_FOR(origin=[0, 0, 0],size=1):
    """
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size,origin=origin)
    mesh_frame.translate(origin)
    return(mesh_frame)


ap = ArgumentParser()
ap.add_argument('-r', '--choose_json', type=int, required=True, help="1-наш json,0-их json")
ap.add_argument('-i', '--input_pcd', type=str, required=True, help="Input .pcd file")
ap.add_argument('-p', '--input_json', type=str, required=True, help="Input .json file")
args = vars(ap.parse_args())

directory="C:/Users/Дмитрий/PycharmProjects/parseJSON"
files = args["input_pcd"]
print(files)
#directory2="C:/Users/Дмитрий/PycharmProjects/parseJSON/clouds_tof_ann"
file_json = args["input_json"]
print(file_json)

#FOR= get_o3d_FOR()

boxes=[]
vis = o3d.visualization.Visualizer()

#проходим по каждому файлу

    #считываем JSON
    #ТУТ МЫ СЧИТЫВАЕМ JSON ФАЙЛ ПОЛНОСТЬЮ
with open( file_json, "r", encoding='utf-8') as fel:
     limb_id=' '
     human_id=' '
     wear_id=' '
     other_id=' '
     text = json.load(fel)
     if args["choose_json"]==0:
        for obj in text["objects"]:
            if obj['classTitle'] == 'human':
                human_id = obj['key']
            if obj['classTitle'] == 'wear':
                wear_id = obj['key']
            if obj['classTitle'] == 'other':
                other_id = obj['key']
            if obj['classTitle'] == 'limb':
                limb_id = obj['key']
pcd = o3d.io.read_point_cloud(files)
        #mybox = my_rect()
READY=[]
if args['choose_json']==1:
        print('cчитаем наш json')
        boxes = get_box1(text)
        lis=''
        if text['events']==[]:
            event='unknown'
        else:
            for event in text['events']:
                lis=lis+'/'+event
        is_can_move=text['is_can_move']
        if is_can_move==False:
            chat='Train can not move'
        else:
            chat='Train can move'
        door_open_proc=str(text['door_open_percent'])
        op=text['door']

        chessboard_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])
       # pcd_1 = text_3d(event, pos=[-1, 0, 0], font_size=350, density=1)
        pcd_1 = text_3d('Door state: ' + op, pos=[-4, 0, 0], font_size=400, density=1)
        if op!='unknown':
            pcd_2 = text_3d('The door is ' + door_open_proc + '% open', pos=[-4, 0, 0.5], font_size=400, density=1)
            pcd_4 = text_3d(chat, pos=[-4, 0, 1.5], font_size=400, density=1)
        else:
            pcd_2 = text_3d(' ', pos=[-4, 0, 0.5], font_size=400, density=1)
            pcd_4 = text_3d(' ', pos=[-4, 0, 2.5], font_size=400, density=1)
        pcd_3 = text_3d('Events: '+lis, pos=[-4, 0, 1.0], font_size=400, density=1)

        #pcd_20 = text_3d('Test-20mm', pos=[0, 0, 0], font_size=20, density=2)
        #ТУТ МЫ СЧИТЫВАЕМ ФАЙЛ ОБЛАКА ТОЧЕК

       # READY.append(FOR)
        READY.append(pcd)
        #READY.append(mybox)
        for j in range(len(boxes)):
            READY.append(boxes[j])
       # READY.append(pcd_1)
        READY.append(pcd_1)
        READY.append(pcd_2)
        READY.append(pcd_3)
        READY.append(pcd_4)
        o3d.visualization.draw_geometries(READY,
                                              width=1024,
                                              height=980)
if args['choose_json']==0:
        print('cчитаем их json')
        boxes = get_box(text)
        READY.append(pcd)
        for j in range(len(boxes)):
            READY.append(boxes[j])
        o3d.visualization.draw_geometries(READY,
                                              width=1024,
                                              height=980)
        print(files)
