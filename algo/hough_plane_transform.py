import numpy as np
import open3d as o3d
import math
from tqdm import tqdm

def _calc_normals(fis, thetas):
    fis_len = len(fis)
    thetas_len = len(thetas)

    normals = np.zeros((fis_len, thetas_len, 3), dtype=np.float)

    fis_rad = fis / 180 * np.pi
    thetas_rad = thetas / 180 * np.pi

    for i in range(fis_len):
        fi = fis_rad[i]
        for j in range(thetas_len):
            theta = thetas_rad[j]
            normal = np.array([
                math.sin(theta) * math.cos(fi),
                math.sin(theta) * math.sin(fi),
                math.cos(theta)
            ])
            normals[i, j] = normal

    return normals

def _dot_prod(point,normals):
    x,y,z = point
    xx,yy,zz = normals[:,:,0], normals[:,:,1], normals[:,:,2]
    dot = x*xx + y*yy + z*zz
    dot = np.abs(dot)
    return dot

def _fi_theta_depth_to_point(fi, theta, depth):
    normal = np.array([
        math.sin(theta) * math.cos(fi),
        math.sin(theta) * math.sin(fi),
        math.cos(theta)
    ])
    return normal * depth

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def hough_planes(points, threshold,
                 fi_step=1, fi_bounds=[0,360], theta_step=1, theta_bounds=[0,180],
                 depth_grads=100, depth_start=0):
    fis = np.arange(fi_bounds[0], fi_bounds[1], fi_step)
    thetas = np.arange(theta_bounds[0], theta_bounds[1], theta_step)

    fis_len = len(fis)
    thetas_len = len(thetas)
    accum = np.zeros([fis_len, thetas_len, depth_grads], dtype=np.int)
    normals = _calc_normals(fis, thetas)

    points_max = np.max(points) * 2
    points_scaled = points * depth_grads / points_max

    fi_idxes = np.zeros([fis_len, thetas_len], dtype=np.int)
    for i in range(len(fis)):
        fi_idxes[i] = i
    fi_idxes = fi_idxes.flatten()
    theta_idxes = np.zeros([fis_len, thetas_len], dtype=np.int)
    for i in range(len(thetas)):
        theta_idxes[:, i] = i
    theta_idxes = theta_idxes.flatten()

    for k in tqdm(range(0, len(points))):
        point = points_scaled[k]

        dists = _dot_prod(point, normals).astype(np.int)
        dists = dists.flatten()

        # for i in range(len(fi_idxes)):
        #     accum[fi_idxes[i], theta_idxes[i], dists[i]] += 1
        accum[fi_idxes, theta_idxes, dists] += 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    points_best = []
    for i in range(len(fis)):
        for j in range(len(thetas)):
            for k in range(depth_start, depth_grads):
                v = accum[i, j, k]
                if v >= threshold:
                    points_best.append([i, j, k, v])
    points_best = np.array(points_best)
    if len(points_best) == 0:
        print('Failed to find hough planes: all points below threshold')
        return None, None


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_best[:,:3])
    cluster_idxes = pcd.cluster_dbscan(eps=3, min_points=5)

    clusters = {}
    for i in range(len(cluster_idxes)):
        idx = cluster_idxes[i]

        if not idx in clusters:
            clusters[idx] = []

        clusters[idx].append(points_best[i])

    planes_out = []
    for k, v in clusters.items():
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        cluster = np.array(v, dtype=np.int)

        coords = cluster[:, :3]
        weights = cluster[:, 3]
        for i in range(3):
            coords[:, i] *= weights
        weight = np.average(weights)

        coord = np.sum(coords, axis=0) / np.sum(weights)
        print('coord', coord, 'weight', weight)

        fi = (fi_bounds[0] + coord[0]*fi_step) / 180 * np.pi
        theta = (theta_bounds[0] + coord[1]*theta_step) / 180 * np.pi
        depth = coord[2] / depth_grads * points_max
        point = _fi_theta_depth_to_point(fi, theta,depth)
        print('fi,theta,depth', fi,theta,depth)
        print('point', point)

        plane = np.concatenate([point, [weight]])
        planes_out.append(plane)
    planes_out = np.array(planes_out)

    return planes_out, points_best