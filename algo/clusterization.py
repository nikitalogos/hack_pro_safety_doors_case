import numpy as np
import math
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from .hough_plane_transform import hough_planes

from .scene_state import SceneState, SceneEvents, SceneObject, ObjectTypes

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~START VIZUALIZATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from plotly.offline import iplot
from plotly import graph_objs as go

def show_points_best(points_best):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points_best[:, 0],
            y=points_best[:, 1],
            z=points_best[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=points_best[:, 3],
                colorbar=dict(
                    thickness=50,
                )
            )
        )
    ])
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=800,
        paper_bgcolor="LightSteelBlue",
    )
    iplot(fig)

def show_points_plane(points, points_plane):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
            )
        ),
        go.Scatter3d(
            x=points_plane[:, 0],
            y=points_plane[:, 1],
            z=points_plane[:, 2],
            mode='markers',
            marker=dict(
                size=5,
            )
        ),
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=10,
                color=(255, 0, 0)
            )
        )
    ])
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=800,
        paper_bgcolor="LightSteelBlue",
    )
    iplot(fig)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def show_points(points):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
            )
        ),
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=10,
                color=(255, 0, 0)
            )
        )
    ])
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=800,
        paper_bgcolor="LightSteelBlue",
    )
    iplot(fig)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def show_points_2d(points):
    fig = go.Figure(data=[
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            marker=dict(
                size=5,
            )
        ),
    ])
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=800,
        paper_bgcolor="LightSteelBlue",
    )
    iplot(fig)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def show_clusters(clusters):
    data = [
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=10,
                color=(255, 0, 0)
            )
        )
    ]
    for k,v in clusters.items():
        random_color = (np.random.random(3)*255).astype(np.uint8)
        color_str = f'rgb({random_color[0]},{random_color[1]},{random_color[2]})'
        print('random_color', random_color, color_str)
        data.append(
            go.Scatter3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color_str,
                )
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=800,
        paper_bgcolor="LightSteelBlue",
    )
    iplot(fig)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def vec_norm(vec):
    x, y, z = vec
    vec_len = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return vec / vec_len

def get_cos(vec1, vec2):
    x, y, z = vec1
    xx, yy, zz = vec2
    dot = x * xx + y * yy + z * zz

    len1 = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    len2 = (xx ** 2 + yy ** 2 + zz ** 2) ** 0.5

    cos = dot / (len1 * len2 + 1e-6)

    return cos

def get_plane_polygon_from_vector(vector, radius):
    orts = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float)

    for ort in orts:
        cos = abs(get_cos(ort, vector))
        if cos < 0.1:
            print(ort, vector, cos)
            continue

        vector1 = np.cross(ort, vector)
        vector2 = np.cross(vector, vector1)
        break

    vector1_norm = vec_norm(vector1)
    vector2_norm = vec_norm(vector2)

    vectors_out = []
    steps = 10
    for i in range(steps):
        direction = np.pi * 2 / steps * i
        vector_out = vector + \
                     math.cos(direction) * vector1_norm * radius + \
                     math.sin(direction) * vector2_norm * radius
        vectors_out.append(vector_out)
    vectors_out.append(vectors_out[0])
    vectors_out = np.array(vectors_out)

    return vectors_out

def visualize_plane(points, vectors):
    data = [
        go.Scatter3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            mode='markers',
            marker=dict(
                size=1,
            )
        ),
        go.Scatter3d(
            x=vectors[:,0],
            y=vectors[:,1],
            z=vectors[:,2],
            mode='markers',
            marker=dict(
                size=5,
            )
        ),
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=10,
                color=(255,0,0)
            )
        )
    ]

    for vec in vectors:
        for radius in [1.0, 2.0, 3.0]:
            poly = get_plane_polygon_from_vector(vec[:3], radius)
            data.append(
                go.Scatter3d(
                    x=poly[:,0],
                    y=poly[:,1],
                    z=poly[:,2],
                    mode='lines',
                ),
            )

    fig = go.Figure(data=data)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=800,
        paper_bgcolor="LightSteelBlue",
    )
    iplot(fig)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~END VIZUALIZATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def find_platform_floor(points, is_plotting=False):
    points_plane, points_best = hough_planes(
        points, 1000,
        fi_step=2, theta_step=0.5,
        theta_bounds=[-10, 10],
        depth_grads=200,
    )
    if points_plane is None:
        return None

    points_plane2 = []
    for plane in points_plane:
        if plane[2] < 0:
            print(f'oops, got inverted vector: {plane}, inverting back to normal')
            plane[:3] *= -1
        points_plane2.append(plane)
    points_plane = np.array(points_plane2)

    if is_plotting:
        show_points_best(points_best)
        show_points_plane(points, points_plane)

    if len(points_plane) > 0:
        best_idx = np.argmax(points_plane[:,3])
        point_plane = points_plane[best_idx]
        return point_plane
    else:
        return None

def find_train_wall(points, is_plotting=False):
    points_plane, points_best = hough_planes(
        points, 1000,
        fi_step=0.5, theta_step=0.5,
        fi_bounds=[80, 100], theta_bounds=[80, 95],
        depth_grads=200, depth_start=3,
    )
    if points_plane is None:
        return None

    points_plane2 = []
    for plane in points_plane:
        if plane[1] > 0:
            print(f'oops, got inverted vector: {plane}, inverting back to normal')
            plane[:3] *= -1
        points_plane2.append(plane)
    points_plane = np.array(points_plane2)

    if is_plotting:
        show_points_best(points_best)
        show_points_plane(points, points_plane)

    if len(points_plane) > 0:
        best_idx = np.argmax(points_plane[:,3])
        point_plane = points_plane[best_idx]
        return point_plane
    else:
        return None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def projection(points,vector):
    vector_norm = vec_norm(vector)

    xx,yy,zz = points[:,0], points[:,1], points[:,2]
    x,y,z = vector_norm
    dot = x*xx + y*yy + z*zz
    return dot

def vec_len(vec):
    x,y,z = vec
    len_ = (x**2 + y**2 + z**2)**0.5
    return len_

def box_tblr(points):
    t, b, l, r = np.max(points[:, 1]), np.min(points[:, 1]), np.min(points[:, 0]), np.max(points[:, 0])
    return t, b, l, r

def points_to_image(points, height=500, point_size=None, is_imshow=False):
    t, b, l, r = box_tblr(points)
    w, h = r - l, t - b

    pad_px = 1
    mul_px = height / h

    image = np.zeros((int(h * mul_px + 2 * pad_px), int(w * mul_px + 2 * pad_px)), dtype=np.uint8)

    # ~~~~~~~~~~~~~
    coords_x = np.around((points[:, 0] - l) * mul_px + pad_px).astype(np.int)
    coords_y = np.around((points[:, 1] - b) * mul_px + pad_px).astype(np.int)

    if point_size is None:
        image[coords_y, coords_x] = 255
    else:
        for i in range(len(points)):
            cv2.circle(image, (coords_x[i], coords_y[i]), point_size, 255, -1)

    if is_imshow:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.show()

    return image

FLOOR_TRIM_DIST_M = 0.1
DOOR_ZONE_WIDTH_M = 0.05

DOOR_STEP_M = 0.1
DOOR_OPEN_THRESHOLD = 100
DOOR_FULL_OPEN_M = 1.2
DOOR_PX_PER_M = 250

MIN_OBJECT_POINTS = 100

def clusterize(points, is_plotting=False) -> object or None:
    if is_plotting:
        show_points(points)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~find floor plane. remove floor points~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pf_vector = find_platform_floor(points, is_plotting=is_plotting)
    if pf_vector is None:
        print('Failed to find_platform_floor!')
        return None
    pf_vector = pf_vector[:3]
    if is_plotting:
        visualize_plane(points, np.expand_dims(pf_vector, axis=0))

    points_proj = projection(points, pf_vector)
    pf_vec_len = vec_len(pf_vector)

    points_mask = points_proj < (pf_vec_len - FLOOR_TRIM_DIST_M)
    points_masked = points[points_mask]

    remove_num = len(points) - len(points_masked)
    print(f'Removed {remove_num} points of floor and beyond')
    points = points_masked

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~find doors plane. select doors points~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tw_vector = find_train_wall(points, is_plotting=is_plotting)
    is_train_wall_found = tw_vector is not None
    if not is_train_wall_found:
        print('Train wall not found!')
    else:
        tw_vector = tw_vector[:3]
        if is_plotting:
            visualize_plane(points, np.expand_dims(tw_vector, axis=0))

        points_proj = projection(points, tw_vector)
        tw_vec_len = vec_len(tw_vector)

        door_mask = (points_proj > (tw_vec_len - DOOR_ZONE_WIDTH_M)) * (points_proj < (tw_vec_len + DOOR_ZONE_WIDTH_M))
        points_door = points[door_mask]
        points_proj_door = points_proj[door_mask]

        #~~~~~~~~
        points_mask = points_proj < (tw_vec_len - DOOR_ZONE_WIDTH_M)
        points_masked = points[points_mask]

        remove_num = len(points) - len(points_masked)
        print(f'Removed {remove_num} points beyond the door')
        points = points_masked

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PERFORM ANALYSIS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PERFORM ANALYSIS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    scene = SceneState()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~start door open percent~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('~~~~~~~~~~~~~~~~~~~~door open percent~~~~~~~~~~~~~~~~~~~~')
    if is_train_wall_found:
        if is_plotting:
            show_points(points_door)

        tw_vector_norm = vec_norm(tw_vector)
        proj_vectors = np.zeros((len(points_proj_door), 3))
        for i in range(3):
            proj_vectors[:, i] = points_proj_door * tw_vector_norm[i]
        door_points_zero_plane = points_door - proj_vectors
        if is_plotting:
            show_points(door_points_zero_plane)

        axis_x = np.cross(tw_vector, pf_vector)
        axis_y = np.cross(tw_vector, axis_x)
        axis_x_norm = vec_norm(axis_x)
        axis_y_norm = vec_norm(axis_y)
        print('axis_x', axis_x)
        print('proj_y', axis_y)

        points_x = projection(door_points_zero_plane, axis_x_norm)
        points_y = projection(door_points_zero_plane, axis_y_norm)

        door_points_2d = np.stack([points_x, points_y], axis=1)

        y_min = np.min(door_points_2d[:,1])
        y_max = y_min + 2
        mask = door_points_2d[:,1] < y_max
        door_points_2d = door_points_2d[mask]

        print('door_points_2d.shape', door_points_2d.shape)

        if is_plotting:
            show_points_2d(door_points_2d)

        image = points_to_image(door_points_2d, height=DOOR_PX_PER_M*2, point_size=8, is_imshow=False)
        h,w = image.shape
        line = np.average(image, axis=0)

        step = int(DOOR_STEP_M*DOOR_PX_PER_M)
        print('step', step)
        bin_bounds = np.arange(0, w, step)
        bins = []
        for i in range(len(bin_bounds) - 1):
            line_segment = line[bin_bounds[i]:bin_bounds[i+1]]
            bin_count = np.average(line_segment)
            bins.append(bin_count)
        bins = np.array(bins)
        bins_is_open = bins < DOOR_OPEN_THRESHOLD
        print('bins', bins)
        print('bins_is_open', bins_is_open)

        max_open_sequence = 0
        this_open_sequence = 0
        for i in range(len(bins_is_open)):
            bin_is_open = bins_is_open[i]
            if bin_is_open:
                this_open_sequence += 1
            else:
                max_open_sequence = max(max_open_sequence, this_open_sequence)
                this_open_sequence = 0
        max_open_sequence = max(max_open_sequence, this_open_sequence)
        door_open_width = max_open_sequence * DOOR_STEP_M
        print('door_open_width', door_open_width)

        door_open_percent = int(door_open_width / DOOR_FULL_OPEN_M * 100)
        door_open_percent = min(door_open_percent, 100)
        scene.set_door_open_percent(door_open_percent)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~end door open percent~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~start detect blobs from remaining points~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('~~~~~~~~~~~~~~~~~~~~detect objects~~~~~~~~~~~~~~~~~~~~')
    if is_plotting:
        show_points(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd_filtered, _ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.25)
    print('pcd_filtered', pcd, pcd_filtered)
    cluster_idxes = pcd_filtered.cluster_dbscan(eps=0.1, min_points=1)
    points = np.asarray(pcd_filtered.points)

    clusters = {}
    for i in range(len(cluster_idxes)):
        idx = cluster_idxes[i]
        if not idx in clusters:
            clusters[idx] = []
        clusters[idx].append(points[i])
    print('clusters.keys()', clusters.keys())

    keys_to_del = []
    for k,v in clusters.items():
        if len(v) < MIN_OBJECT_POINTS:
            keys_to_del.append(k)
            continue
        clusters[k] = np.array(clusters[k])
    for k in keys_to_del:
        del clusters[k]

    print('clusters.keys()', clusters.keys())

    if is_plotting:
        show_clusters(clusters)

    for k,v in clusters.items():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v[:, :3])
        mins = pcd.get_min_bound()
        maxes = pcd.get_max_bound()

        position = (mins + maxes) / 2
        dimensions = maxes - mins

        obj = SceneObject(
            object_type=ObjectTypes.HUMAN,
            position=position,
            dimensions=dimensions,
        )
        scene.add_object(obj)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~end detect blobs from remaining points~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~start compute events~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if is_train_wall_found:
        for obj in scene.objects:
            vertices = obj.get_box_vertices()
            vertices_proj = projection(vertices, tw_vector_norm)
            if np.sum(vertices_proj > tw_vec_len) > 0:
                print('Detected object inside door!')
                scene.add_event(SceneEvents.OBJECT_BETWEEN_DOORS)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~end compute events~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return scene



