#! /usr/bin/python3

from argparse import ArgumentParser
import numpy as np
import open3d as o3d

from clusterization import clusterize

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-i', '--input', type=str, required=True)
    args = vars(ap.parse_args())

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

    clusterize(points, is_plotting=False)