import numpy as np
import math
from hough_plane_transform import hough_planes

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
        fi_step=0.5, theta_step=0.5,
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
        point_plane = np.average(points_plane, axis=0)
        return point_plane
    else:
        return None

def find_train_wall(points, is_plotting=False):
    points_plane, points_best = hough_planes(
        points, 500,
        fi_step=0.5, theta_step=0.5,
        fi_bounds=[80, 100], theta_bounds=[80, 100],
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
        point_plane = np.average(points_plane, axis=0)
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

FLOOR_TRIM_DIST_M = 0.1
DOOR_ZONE_WIDTH_M = 0.05

def clusterize(points, is_plotting=False):
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
    if tw_vector is None:
        print('Failed to find_train_wall!')
        return None
    tw_vector = tw_vector[:3]
    if is_plotting:
        visualize_plane(points, np.expand_dims(tw_vector, axis=0))

    points_proj = projection(points, tw_vector)
    tw_vec_len = vec_len(tw_vector)

    door_mask = (points_proj > (tw_vec_len - DOOR_ZONE_WIDTH_M)) & (points_proj < (tw_vec_len + DOOR_ZONE_WIDTH_M))
    points_door = points[door_mask]

    #~~~~~~~~
    points_mask = points_proj < (pf_vec_len + DOOR_ZONE_WIDTH_M)
    points_masked = points[points_mask]

    remove_num = len(points) - len(points_masked)
    print(f'Removed {remove_num} points beyond the door')
    points = points_masked

    # detect blobs from remaining points
    if is_plotting:
        show_points(points)
        show_points(points_door)

