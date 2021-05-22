#!/bin/sh
"exec" "`dirname $0`/venv/bin/python" "$0" "$@"
#ГОЛУБОЕ-ЧЕЛОВЕК
#КРАСНОЕ-КОНЕЧНОСТЬ
#ЗЕЛЕНОЕ-ДРУГОЕ
#ЧЕРНОЕ-ОДЕЖДА

import open3d as o3d
import json
import os
def fun ():
    return
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
                points = [[init_cord[0] - disp[0] / 2, init_cord[1], init_cord[2]],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1], init_cord[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1], init_cord[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1], init_cord[2]],

                          [init_cord[0] - disp[0] / 2, init_cord[1], init_cord[2] - disp[2]],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1], init_cord[2] - disp[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1], init_cord[2] - disp[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1], init_cord[2] - disp[2]],
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
                points = [[init_cord[0] - disp[0] / 2, init_cord[1], init_cord[2]],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1], init_cord[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1], init_cord[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1], init_cord[2]],

                          [init_cord[0] - disp[0] / 2, init_cord[1], init_cord[2] - disp[2]],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1], init_cord[2] - disp[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1], init_cord[2] - disp[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1], init_cord[2] - disp[2]],
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
                points = [[init_cord[0] - disp[0] / 2, init_cord[1], init_cord[2]],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1], init_cord[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1], init_cord[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1], init_cord[2]],

                          [init_cord[0] - disp[0] / 2, init_cord[1], init_cord[2] - disp[2]],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1], init_cord[2] - disp[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1], init_cord[2] - disp[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1], init_cord[2] - disp[2]],
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
                points = [[init_cord[0] - disp[0] / 2, init_cord[1], init_cord[2]],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1], init_cord[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1], init_cord[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1], init_cord[2]],

                          [init_cord[0] - disp[0] / 2, init_cord[1], init_cord[2] - disp[2]],
                          [init_cord[0] - disp[0] / 2, init_cord[1] + disp[1], init_cord[2] - disp[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1] + disp[1], init_cord[2] - disp[2]],
                          [init_cord[0] + disp[0] / 2, init_cord[1], init_cord[2] - disp[2]],
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
def my_rect():
    points = [[0, 0, 0],
              [1, 0, 0],
              [1, 1, 0],
              [0,1, 0],

              [0, 0, 5],
              [1, 0, 5],
              [1, 1, 5],
              [0, 1, 5],
              ]
    lines = [[0, 1], [0, 3], [1, 2], [2, 3],
             [4, 5], [4, 7], [5, 6], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[0, 1, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set



directory1="C:/Users/Дмитрий/PycharmProjects/parseJSON/points"
files = os.listdir(directory1)
print(files)
directory2="C:/Users/Дмитрий/PycharmProjects/parseJSON/clouds_tof_ann"
file_json = os.listdir(directory2)

human_id="6bc1d79329bd444c8fb0717e408dd48c"
wear_id="f8b0fcbf2fd140caa49049729ad18072"
other_id="dc58658c7ffd4802a1dd62258b4ad985"
limb_id="e7eb42dd5e0544eb9f667a3915a64700"

FOR= get_o3d_FOR()

boxes=[]
vis = o3d.visualization.Visualizer()

#проходим по каждому файлу
for i in range(len(files)):
    #считываем JSON
    with open(os.path.join(directory2, file_json[i]), "r", encoding='utf-8') as fel:
        text = json.load(fel)
        boxes=get_box(text)
        #mybox = my_rect()
        READY=[]

        pcd = o3d.io.read_point_cloud(os.path.join("points/", files[i]))
        READY.append(FOR)
        READY.append(pcd)
        #READY.append(mybox)
        for j in range(len(boxes)):
            READY.append(boxes[j])
        o3d.visualization.draw_geometries(READY,
                                              width=1024,
                                              height=980)
        print(files[i])

