import os
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
from projection_utils import get_axismatrix, projection_to_rgb

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Change these variables if needed.
ROOT_DIR = '/home/chidubem/deep-3d-reconstruction/scannet/scans/'
BATCH_SIZE = 2

HEIGHT = 968
WIDTH = 1296
TRAIN_SIZE = 15
VIEWS_SIZE = 5


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
    """Serialize to tensorflow example and save to tf records.
    Args:
        images (np.float32): array containing multiple rgb images for each scene.
        depths (np.float32): array containing multiple  depth images for each scene.
        extrinsic_matrix (np.float64): (Mx4x4 array): multi array of camera to world extrinsic parameters.
        intrinsic_matrix (np.float64): (4x4 array): camera intrinsic parameters.
        axis_alinment (np.float32): (4x4 array): axis_alignment matrix.
        box_objects (np.float64): Nx7 array containing box center, size and angle of N boxes.
    Return: serialized tensorflow example of a scene.
    """

    feature = {
        'images': _bytes_feature(images),
        'depths': _bytes_feature(depths),
        'extrinsic_matrix': _bytes_feature(extrinsic_matrix),
        'intrinsic_matrix': _bytes_feature(intrinsic_matrix),
        'axis_matrix': _bytes_feature(axis_matrix),
        'box_objects': _bytes_feature(box_objects),
    }
    #  Create a message using tf.train.Example.
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
    """Saves tensoflow examples to tf records.
    Args:
        image_paths: array of paths of rgb images for each scene.
        depth_paths: array of paths of depth images for each scene.
        extrinsic_matrix_paths: array of paths for camera to world extrinsic parameters.
        intrinsic_matrix: path for camera intrinsic parameters.
        axis_alinment: path for axis_alignment matrix.
        box_objects: path for box objects.
        filename: path to save the tfrecord.
    Return: serialized tensorflow example of a scene.
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
            images.append(img_array)

            depth_img = mpimg.imread(depth_path)
            depths.append(depth_img)

            extrinsics.append(np.loadtxt(extrinsic_path))

        images = tf.io.serialize_tensor(images)
        depths = tf.io.serialize_tensor(depths)
        extrinsics = tf.io.serialize_tensor(extrinsics)
        intrinsic_matrix = tf.io.serialize_tensor(
            np.loadtxt(intrinsic_matrix_path))
        axis_matrix = tf.io.serialize_tensor(get_axismatrix(axis_matrix_path))
        box_objects = str.encode(box_objects_path)
        # box_objects = tf.io.serialize_tensor(np.loadtxt(box_objects_path))

        example = serialize_example(
            images,
            depths,
            extrinsics,
            intrinsic_matrix,
            axis_matrix,
            box_objects)
        writer.write(example)


def read_decode_tfrecords(serialized_example):
    """Reads and parses tfrecords.
    Args:
        serialized_example: path to save the tfrecord.
    Return: dictionary of data pipeline.
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
    example['box_objects'] = example['box_objects']
#     example['box_objects'] = tf.io.parse_tensor(
#         example['box_objects'], out_type=tf.float64)

    # Linearly scales each image in images to have mean 0 and variance 1.
#     example['images'] = tf.map_fn(
#         lambda image: tf.image.per_image_standardization(image),
#         example['images'])
    
    # scale to [0,1], either use above normalization or this
    example['images'] = example['images']/255
        
    # for depth and pose estimation model
    example['img'] =  example['images'][0]
    example['img-1'] = example['images'][1]
    example['img-2'] = example['images'][2]
    example['img1'] = example['images'][3]
    example['img2'] = example['images'][4]
    example['K'] = tf.cast(example['intrinsic_matrix'], tf.float32)
    example['K_inv'] = tf.linalg.inv(example['K'])

    return example


def get_images_with_boxes(scene_dir):
    """Get selected random images that have most boxes.
    Args:
        scene_dir: directory path for scene.
    Return: image paths, depth_paths and extrinsic paths.
    """
    seed = 2000
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

        # Check if center of box in image.
        count = 0
        for i, b in enumerate(boxes):
            b = np.float32(b)
            x, y = b[:, 0], b[:, 1]
            if (x[0] < WIDTH) and (y[0] < HEIGHT):
                count += 1
        center_counts.append(count)

    # Sort in center counts in descending order and select the top images.
    center_indexes = np.argsort(center_counts)[::-1][:TRAIN_SIZE+5]
    assert center_indexes.size >= TRAIN_SIZE, "reduce train size"
    # Randomly select TRAIN_SIZE images from the list.
    np.random.seed(seed)
    center_indexes = np.random.choice(
        center_indexes, TRAIN_SIZE, replace=False)

    image_paths = [os.path.join(image_paths, images[i])
                   for i in center_indexes]
    depth_paths = [os.path.join(depth_paths, images[i][:-3] + 'png')
                   for i in center_indexes]
    extrinsic_paths = [os.path.join(
        extrinsic_paths, images[i][:-3] + 'txt') for i in center_indexes]

    return image_paths, depth_paths, extrinsic_paths


def get_views_from_images(scene_dir):
    """For each scene, get random views of the same object.
    Args:
        scene_dir: directory path for scene.
    Return: image paths, depth_paths and extrinsic paths.
    """
    seed = 2000
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

    images = [image for image in os.listdir(image_paths) if image.endswith('.jpg')]
    image_views_paths = []
    depth_views_paths = []
    extrinsic_views_paths = []
    
    for box in range(box_objects.shape[0]):
        box_images = []
        for image in images:
            extrinsic_matrix = np.loadtxt(os.path.join(
                extrinsic_paths, image[:-3] + 'txt'))
            boxes = projection_to_rgb(
                box_objects,
                intrinsic_matrix,
                extrinsic_matrix,
                axis_matrix)

            # Check if center of box in image.
            b = np.float32(boxes[box])
            x, y = b[:, 0], b[:, 1]
            if (x[0] < WIDTH) and (y[0] < HEIGHT):
                box_images.append(image)
        
        # randomly select only views_size images
        np.random.seed(seed)        
        box_images = np.random.choice(box_images, VIEWS_SIZE, replace=False)

        image_views_paths = image_views_paths + [os.path.join(image_paths, box_images[i])
                       for i in range(VIEWS_SIZE)]
        depth_views_paths = depth_views_paths + [os.path.join(depth_paths, box_images[i][:-3] + 'png')
                       for i in range(VIEWS_SIZE)]
        extrinsic_views_paths = extrinsic_views_paths + [os.path.join(
            extrinsic_paths, box_images[i][:-3] + 'txt') for i in range(VIEWS_SIZE)]
    

    return image_views_paths, depth_views_paths, extrinsic_views_paths


def get_scene_data(scenename, train_data='tfdata'):
    """Reads input data from scene. 
    The sens file should be parsed for each scene and box_coords should contain center, size and heading angles of all objects in scene.
    Args:
        scenename: name of scene.
    Return: inputs for data pipeline.
    """
    scene_dir = os.path.join(ROOT_DIR, scenename)
    # image_paths, depth_paths, extrinsic_matrix_paths = get_images_with_boxes(scene_dir)
    image_paths, depth_paths, extrinsic_matrix_paths = get_views_from_images(scene_dir)
    intrinsic_matrix_path = os.path.join(scene_dir, 'intrinsic_color.txt')
    axis_matrix_path = os.path.join(scene_dir, scenename + '.txt')
    box_objects_path = os.path.join(scene_dir, 'box_coords.txt')
    filename = os.path.join(train_data, scenename + '.tfrecords')

    return image_paths, depth_paths, extrinsic_matrix_paths, intrinsic_matrix_path, axis_matrix_path, box_objects_path, filename


def write_data_to_tfrecords(train_data):
    """Saves data for training as tfrecords."""
    assert os.path.isfile(train_data)
    for line in open(train_data):
        # save_to_tfrecords(*get_scene_data(line.strip()))
        count = 0
        image_paths, depth_paths, extrinsic_matrix_paths, intrinsic_matrix_path, axis_matrix_path, \
        box_objects_path, filename = get_scene_data(line.strip(),'train_data')
        number_of_boxes = int(len(image_paths)/VIEWS_SIZE)
        filename = filename.split('.')
        for i in range(number_of_boxes):
            # added box_id to box path
            save_to_tfrecords(image_paths[count:count+VIEWS_SIZE], depth_paths[count:count+VIEWS_SIZE], \
                              extrinsic_matrix_paths[count:count+VIEWS_SIZE], \
                              intrinsic_matrix_path, axis_matrix_path, box_objects_path[:-4]+str(i)+'.txt', \
                              filename[0]+str(i)+'.'+filename[1])
            count += VIEWS_SIZE

def input_fn(tfrecords, map_fn=None, load_option='train'):
    """Create input pipeline for the model. Loads all dataset
    Args:
        tfrecords: file lists or file of tfrecords.
    Return: dataset for training.
    """

    dataset = tf.data.TFRecordDataset(tfrecords)
    if map_fn is None:
        dataset = dataset.map(read_decode_tfrecords, num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
    if load_option=='train':
        # Ensure box objects have same size when using batch.
        dataset = dataset.shuffle(400).repeat(BATCH_SIZE).batch(BATCH_SIZE)
    else:
        dataset = dataset.shuffle(400).batch(1)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
