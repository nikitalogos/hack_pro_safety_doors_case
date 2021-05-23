#!/bin/sh
"exec" "`dirname $0`/venv/bin/python" "$0" "$@"

from argparse import ArgumentParser
import numpy as np
import open3d as o3d
import json
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from clusterization import clusterize

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-i', '--input', type=str, required=True, help="Input .pcd file")
    ap.add_argument('-o', '--output', type=str, help='Output json file')
    ap.add_argument('-p', '--is_plotting', action="store_true", help='Show debug plots')
    args = vars(ap.parse_args())
    print('args', args)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    points = []
    colors = []
    with open(args['input'], 'r') as inf:
        lines = inf.readlines()
        for line in lines[11:]:
            values = line.strip().split(' ')
            point = [float(i) for i in values[:3]]
            color = int(values[3])
            color = [(color // 2 ** 24) & 255, (color // 2 ** 16) & 255, (color // 2 ** 8) & 255, color & 255]
            points.append(point)
            colors.append(color)

    points_arr = np.array(points)
    colors_arr = np.array(colors, dtype=np.float) / 255
    print(f'Loaded point cloud with {len(points_arr)} points')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_arr)
    pcd.colors = o3d.utility.Vector3dVector(colors_arr[:, :3])

    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    points = np.asarray(pcd.points)
    print(f'Applied filtering. {len(points)} points remaining')

    scene = clusterize(points, is_plotting=args['is_plotting'])
    if scene is None:
        print('SCENE ANALYSIS FAILED!')
        exit(1)

    data_json = scene.to_json()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SCENE STATE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    pp.pprint(data_json)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    file_out = args['output']
    if file_out is not None:
        with open(file_out, 'w') as outf:
            json.dump(data_json, outf)
        print(f'Wrote results to {file_out}')