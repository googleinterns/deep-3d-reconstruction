from __future__ import division
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData
import numpy as np
import cv2

EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)

FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane = front for scannet dataset
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])


def get_axismatrix(axisfile):
    """Process axis alignment matrix.
        axisfile: file containing axis alignment matrix.
        Return: 4x4 axis alignment matrix.
    """
    f = open(axisfile)
    axis_align_matrix = np.array(
        f.readline().strip().split(" ")[
            2:]).astype(np.float32)
    axis_align_matrix = np.reshape(axis_align_matrix, (4, 4))

    return axis_align_matrix


def read_mesh_vertices(filename):
    """ Read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
    return vertices


def align_axis(axis_file, ply_file, sample=False):
    """Align the point clouds using the axis alignment matrix.
        Args:
          axis_file: file containig the axis alignment matrix.
          ply_file: point cloud file.
          sample: boolean value to sample from point clouds.

        Return: x,y,z numpy array of 5000 points if sample is true.
    """
    assert os.path.isfile(axis_file)
    axis_align_matrix = get_axismatrix(axis_file)
    mesh_vertices = read_mesh_vertices(ply_file)

    # Align the vertices to axis co-ordinate.
    points = np.ones((mesh_vertices.shape[0], 4))
    points[:, 0:3] = mesh_vertices[:, 0:3]
    # Rigid body transformation of the points.
    points = np.matmul(axis_align_matrix, points.transpose()).T
    mesh_vertices[:, 0:3] = points[:, 0:3]

    if sample:
        points = np.arange(points.shape[0])
        points = np.random.choice(points, 5000, replace=False)

    return mesh_vertices[points]


def rotation_matrix(heading_angle):
    """Rotate axis by heading angle.

        heading_angle: angle of rotation.
        Return: 3x3 rotation matrix.
    """
    rotmat = np.zeros((3, 3))
    rotmat[2, 2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotmat


def local_coord(scale):
    """Define a 3D axis-aligned coord cube centered at 0.

        scale: size 3 list/array for size of the cube.
        Return: 9x3 array of local vertices for cube with center at 0.
    """
    w = scale[0] / 2.
    h = scale[1] / 2.
    d = scale[2] / 2.

    # Define the local coordinate system, w.r.t. the center of the box.
    box_coords = np.array([[0., 0., 0.], [-w, -h, -d], [-w, -h, +d], [-w, +h, -d],
                           [-w, +h, +d], [+w, -h, -d], [+w, -h, +d], [+w, +h, -d],
                           [+w, +h, +d]])
    return box_coords


def gen_vertices(box_points, heading_angle):
    """Generate homogenous vertices of bounding boxes given center, size and heading angle of box.
        Args:
          box_points: size 7 array containing box center, size and angle in degrees.
          heading_angle: angle of rotation.

        Return: 9x4 array of vertices for bounding box.
    """
    # Convert angle to radians.
    heading_angle = heading_angle * np.pi / 180

    center = box_points[:3]
    lengths = box_points[3:6]
    transformations = np.eye(4)
    transformations[0:3, 3] = center
    transformations[3, 3] = 1.0
    transformations[0:3, 0:3] = rotation_matrix(heading_angle)

    # Returns box center and has homogenous coord.
    local_coords = local_coord(lengths)
    local_coords = np.c_[local_coords, np.ones(local_coords.shape[0])]
    box_vertices = np.matmul(trns, local_coords.T).T
    return box_vertices


def get_box_vertices(box_objects, heading_angles=None):
    """Generate homogenous vertices of bounding boxes given center, size and heading angle of box.
        Args:
          box_objects: Nx7 array containing box center, size and angle of N boxes.
          heading_angles N angles.

        Return: Nx9x3 array of vertices for bounding box.
    """

    # Calculate 3d-vertices of objects in image.
    num_objects = box_objects.shape[0]
    box_vertices = np.zeros((num_objects, 9, 4))
    for i in range(num_objects):
        heading_angle = box_objects[i][-1] if heading_angles is None else heading_angles[i]
        box_vertices[i, :, :] = gen_vertices(box_objects[i], heading_angle)

    return box_vertices


def convert2camera_coords(box_vertices, extrinsic_matrix):
    """Calculate camera coords of the 3d box.
    Args:
        box_vertices: (3-dimensional array) vertices of all objects in the 3d scene.
        extrinsic_matrix: (4x4 array): camera extrinsic parameters.
    Return: camera coordinate (3-dimensional array) of all objects.
    """
    box_vertices = np.transpose(box_vertices, (0, 2, 1))
    # Convert to 2d camera co-ordinates.
    camera_coords = np.zeros(box_vertices.shape)
    for i in range(camera_coords.shape[0]):
        camera_coords[i] = np.matmul(extrinsic_matrix, box_vertices[i])

    return np.transpose(camera_coords, (0, 2, 1))


def convert2pixel_coords(camera_coordinates, intrinsic_matrix):
    """Calculate pixel coords of the 3d box.
    Args:
        camera_coordinates: (3-dimensional array) camera coords of all objects in the 3d scene.
        intrinsic_matrix: (4x4 array): camera intrinsic parameters.
    Output: pixel points in 2D coordinate for all objects.
    """

    camera_coordinates = np.transpose(camera_coordinates, (0, 2, 1))
    pixel_coords = np.zeros(camera_coordinates.shape)
    for i in range(pixel_coords.shape[0]):
        pixel_coords[i] = np.matmul(intrinsic_matrix, camera_coordinates[i])
        # Project 3D points to 2D.
        pixel_coords[i, :3, :] = np.nan_to_num(
            pixel_coords[i, :3, :] / pixel_coords[i, 2, :])

    return np.round(np.transpose(pixel_coords, (0, 2, 1)))


def create_aligned_pose(axis_alignment, extrinsic_matrix):
    """Axis align the extrinsic matrix and convert to world to camera coords.
    Args:
        axis_alinment: axis_alignment matrix.
        extrinsic_matrix: (4x4 array): camera to world extrinsic parameters.
    Return: axis aligned world to camera pose.
    """
    pose = np.matmul(axis_alignment, extrinsic_matrix)
    # Invert the pose since the extrinsic matrix is camera to world. 
    pose = np.linalg.inv(pose)

    return pose


def projection_to_rgb(box_objects, intrinsic_matrix, extrinsic_matrix,
                      axis_alignment, align_pose=True, heading_angles=None):
    """ Project points from 3D to 2D images.
    Args:
        box_objects: Nx7 array containing box center, size and angle of N boxes.
        intrinsic_matrix: (4x4 array): camera intrinsic parameters.
        extrinsic_matrix: (4x4 array): camera extrinsic parameters.
        axis_alinment: axis_alignment matrix.
        align pose: Boolean value set to True is extrinsic matrix is camera to world and has to be axis aligned.
        heading_angles: angle heading for each box object.

    Output: pixel points in 2D coordinate for all objects.
    """
    if align_pose:
        extrinsic_matrix = create_aligned_pose(
            axis_alignment, extrinsic_matrix)
    box_vertices = get_box_vertices(box_objects, heading_angles)
    camera_coordinates = convert2camera_coords(box_vertices, extrinsic_matrix)
    pixel_coordinates = convert2pixel_coords(
        camera_coordinates, intrinsic_matrix)

    return pixel_coordinates


def draw_boxes(boxes=[], infile="", axisfile="", colors=['r', 'b', 'g', 'k'], name='verts.png'):
    """Draw a list of boxes.

      The boxes are defined as a list of vertices
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    if infile != "":
        verts = align_axis(axisfile, infile, sample=True)
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='r', marker='o')

    ax.set_xlabel('X Coord')
    ax.set_ylabel('Y Coord')
    ax.set_zlabel('Z Coord')

    for i, b in enumerate(boxes):
        x, y, z = b[:, 0], b[:, 1], b[:, 2]
        ax.scatter(x, y, z, c='r')
        for e in EDGES:
            ax.plot(x[e], y[e], z[e], linewidth=2, c=colors[i % len(colors)])

    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))

    # Rotate the axes and update.
    ax.view_init(30, 12)
    plt.draw()
    plt.savefig(name)
    plt.show()


def draw_2dboxes(filename, boxes=[], points=True, greyscale=False, flip=True):
    """Draw a list of boxes.
    Args:
        filename: file path to the image.

      The boxes are defined as a list of vertices.
    """
    def make_diag_edges(face):
        diag_a = face[[0, 2]]
        diag_b = face[[1, 3]]

        return diag_a, diag_b

    fig = plt.figure(figsize=(10, 10))

    if greyscale:
        cvimg = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        cvimg = cv2.imread(filename)
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)

    height, width = cvimg.shape[:2]

    if flip:
        cvimg = cv2.flip(cvimg, 0)

    for i, b in enumerate(boxes):
        b = np.float32(b)
        x, y = b[:, 0], b[:, 1]

        # Skip points whose center is outside of the image.
        if (x[0] > width) or (y[0] > height):
            continue

        # Plot box front.
        diag_a, diag_b = make_diag_edges(FACES[0])

        # Plot the center.
        if points:
            if flip:
                plt.scatter(x, height - y, c='r')
                plt.plot(x[diag_a], height - y[diag_a], c='b')
                plt.plot(x[diag_b], height - y[diag_b], c='b')
                if (0 <= x[0] <= width) and (0 <= y[0] <= height):
                    plt.text(x[0],
                             height - y[0],
                             'Box {}'.format(str(i + 1)),
                             color='white')
            else:
                plt.scatter(x, y, c='r')
                plt.plot(x[diag_a], y[diag_a], c='b')
                plt.plot(x[diag_b], y[diag_b], c='b')
                if (0 <= x[0] <= width) and (0 <= y[0] <= height):
                    plt.text(x[0], y[0], 'Box {}'.format(
                        str(i + 1)), color='white')

        # Plot the edges.
        for e in EDGES:
            if points:
                if flip:
                    plt.plot(x[e], height - y[e])

                else:
                    plt.plot(x[e], y[e])

    plt.xlim([0, width])
    plt.ylim([0, height])
    plt.imshow(cvimg, origin='lower')
    plt.show()
