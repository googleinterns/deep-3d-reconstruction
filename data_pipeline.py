import tensorflow as tf
import numpy as np
from projection_utils import get_axismatrix, projection_to_rgb
import os

ROOT_DIR = '/home/chidubem/deep-3d-reconstruction/scannet/scans/'
BATCH_SIZE = 5
NUM_EPOCHS = 5


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(
        images,
        depths,
        extrinsic_matrix,
        intrinsic_matrix,
        axis_matrix,
        box_objects):
    """Serialize to tensoflow example and save to tf records
    Args:
        images: array of rgb images for each scene
        depths: array of depth images for each scene
        extrinsic_matrix: (4x4 array): camera to world extrinsic parameters
        intrinsic_matrix: (4x4 array): camera intrinsic parameters
        axis_alinment: (4x4 array): axis_alignment matrix
        box_objects: Nx7 array containing box center, size and angle of N boxes
    Return: serialized tensorflow example of a scene
    """

    feature = {
        'images': _bytes_feature(images),
        'depths': _bytes_feature(depths),
        'extrinsic_matrix': _bytes_feature(extrinsic_matrix),
        'intrinsic_matrix': _bytes_feature(intrinsic_matrix),
        'axis_matrix': _bytes_feature(axis_matrix),
        'box_objects': _bytes_feature(box_objects),
    }

    #  Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(
        features=tf.train.Features(
            feature=feature))
    return example_proto.SerializeToString()


def save_to_tfrecords(
        image_paths,
        depth_paths,
        extrinsic_matrix_paths,
        intrinsic_matrix_path,
        axis_matrix_path,
        box_objects_path,
        filename):
    """Saves tensoflow examples to tf records
    Args:
        image_paths: array of paths of rgb images for each scene
        depth_paths: array of paths of depth images for each scene
        extrinsic_matrix_paths: array of paths for camera to world extrinsic parameters
        intrinsic_matrix: path for camera intrinsic parameters
        axis_alinment: path for axis_alignment matrix
        box_objects: path for box objects
        filename: path to save the tfrecord
    Return: serialized tensorflow example of a scene
    """

    images = []
    depths = []
    extrinsics = []
    assert len(image_paths) == len(depth_paths) == len(extrinsic_matrix_paths)
    with tf.io.TFRecordWriter(filename) as writer:
        for image_path, extrinsic_path, depth_path in zip(
                image_paths, extrinsic_matrix_paths, depth_paths):

            img = tf.keras.preprocessing.image.load_img(image_path)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array
            images.append(img_array)

            img = tf.keras.preprocessing.image.load_img(depth_path)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array
            depths.append(img_array)

            extrinsics.append(np.loadtxt(extrinsic_path))

        images = tf.io.serialize_tensor(images)
        depths = tf.io.serialize_tensor(depths)
        extrinsics = tf.io.serialize_tensor(extrinsics)
        intrinsic_matrix = tf.io.serialize_tensor(
            np.loadtxt(intrinsic_matrix_path))
        axis_matrix = tf.io.serialize_tensor(get_axismatrix(axis_matrix_path))
        box_objects = tf.io.serialize_tensor(np.loadtxt(box_objects_path))

        example = serialize_example(
            images,
            depths,
            extrinsics,
            intrinsic_matrix,
            axis_matrix,
            box_objects)
        writer.write(example)


def read_decode_tfrecords(serialized_example):
    """Reads and parses tfrecords
    Args:
        serialized_example: path to save the tfrecord
    Return: dictionary of data pipeline
    """
    feature_description = {
        'images': tf.io.FixedLenFeature((), tf.string),
        'depths': tf.io.FixedLenFeature((), tf.string),
        'extrinsic_matrix': tf.io.FixedLenFeature((), tf.string),
        'intrinsic_matrix': tf.io.FixedLenFeature((), tf.string),
        'axis_matrix': tf.io.FixedLenFeature((), tf.string),
        'box_objects': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(
        serialized_example, feature_description)

    example['images'] = tf.io.parse_tensor(
        example['images'], out_type=tf.float32)
    example['depths'] = tf.io.parse_tensor(
        example['depths'], out_type=tf.float32)
    example['extrinsic_matrix'] = tf.io.parse_tensor(
        example['extrinsic_matrix'], out_type=tf.float64)
    example['intrinsic_matrix'] = tf.io.parse_tensor(
        example['intrinsic_matrix'], out_type=tf.float64)
    example['axis_matrix'] = tf.io.parse_tensor(
        example['axis_matrix'], out_type=tf.float32)
    example['box_objects'] = tf.io.parse_tensor(
        example['box_objects'], out_type=tf.float64)

    # Linearly scales each image in images to have mean 0 and variance 1.
    example['images'] = tf.map_fn(
        lambda image: tf.image.per_image_standardization(image),
        example['images'])

    return example


def get_images_with_boxes(scene_dir):
    """Get selected random images that have most boxes
    Args:
        scene_dir: directory path for scene
    Return: image paths, depth_paths and extrinsic paths
    """

    image_paths = os.path.join(scene_dir, 'color')
    depth_paths = os.path.join(scene_dir, 'depth')
    extrinsic_paths = os.path.join(scene_dir, 'pose')

    scene = os.path.basename(scene_dir)
    axis_matrix = get_axismatrix(os.path.join(scene_dir, scene + '.txt'))
    intrinsic_matrix = np.loadtxt(
        os.path.join(
            scene_dir,
            'intrinsic_color.txt'))
    box_objects = np.loadtxt(os.path.join(scene_dir, 'box_coords.txt'))

    # assuming all images have the same width and height, change if not true
    height = 968
    width = 1296
    train_size = 15

    images = os.listdir(image_paths)
    center_counts = []
    for image in images:
        if not image.endswith('.jpg'):
            center_counts.append(0)
            continue
        extrinsic_matrix = np.loadtxt(os.path.join(
            extrinsic_paths, image[:-3] + 'txt'))
        boxes = projection_to_rgb(
            box_objects,
            intrinsic_matrix,
            extrinsic_matrix,
            axis_matrix)

        # check if center of box in image
        count = 0
        for i, b in enumerate(boxes):
            b = np.float32(b)
            x, y = b[:, 0], b[:, 1]
            if (x[0] < width) and (y[0] < height):
                count += 1
        center_counts.append(count)

    # sort in center counts in descending order and select the top 30 images
    center_indexes = np.argsort(center_counts)[::-1][:30]
    assert center_indexes.size >= train_size, "reduce train size"
    # randomly select train_size images from the list
    np.random.seed(2000)
    center_indexes = np.random.choice(
        center_indexes, train_size, replace=False)

    image_paths = [os.path.join(image_paths, images[i])
                   for i in center_indexes]
    depth_paths = [os.path.join(depth_paths, images[i][:-3] + 'png')
                   for i in center_indexes]
    extrinsic_paths = [os.path.join(
        extrinsic_paths, images[i][:-3] + 'txt') for i in center_indexes]

    return image_paths, depth_paths, extrinsic_paths


def get_scene_data(scenename):
    """Reads input data from scene
    Args:
        scenename: name of scene
    Return: inputs for data pipeline
    """
    scene_dir = os.path.join(ROOT_DIR, scenename)
    image_paths, depth_paths, extrinsic_matrix_paths = get_images_with_boxes(
        scene_dir)
    intrinsic_matrix_path = os.path.join(scene_dir, 'intrinsic_color.txt')
    axis_matrix_path = os.path.join(scene_dir, scenename + '.txt')
    box_objects_path = os.path.join(scene_dir, 'box_coords.txt')
    filename = os.path.join('tfdata', scenename + '.tfrecords')

    return image_paths, depth_paths, extrinsic_matrix_paths, intrinsic_matrix_path, axis_matrix_path, box_objects_path, filename


def write_data_to_tfrecords(train_data):
    """Saves data for training as tfrecords"""
    assert os.path.isfile(train_data)
    count = 0
    for line in open(train_data):
        count += 1
        save_to_tfrecords(*get_scene_data(line.strip()))


def input_fn(tfrecords):
    """Create input pipeline for the model
    Args:
        tfrecords: file lists or file of tfrecords
    Return: dataset for training
    """

    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.map(read_decode_tfrecords, num_parallel_calls=4)
# dataset = dataset.shuffle(400).repeat().batch(BATCH_SIZE) # box objects
# have different shape so cannot be batched
    dataset = dataset.shuffle(400).repeat(NUM_EPOCHS)
    dataset = dataset.prefetch(buffer_size=10)

    return dataset
