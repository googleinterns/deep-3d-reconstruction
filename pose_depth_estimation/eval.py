import argparse
import glob
import os, sys
import pathlib

import cv2
import numpy as np
import tensorflow as tf
from disparitynet import DisparityNet
from utils import readlines, generate_depth_map, visualize_colormap, compute_errors, disp_to_depth

parser = argparse.ArgumentParser(description="Disparity Project")
parser.add_argument('--identifier', default="sfm_resnet18")
parser.add_argument('--dataset_dir', default='../train_data')
parser.add_argument('--demo_set', default="../scannet/scans")
parser.add_argument("--input_h", default=480)
parser.add_argument("--input_w", default=640)

PROJECT_DIR = os.getcwd()
HOME = str(pathlib.Path.home())

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

BASE_DIR = os.path.dirname(PROJECT_DIR)
sys.path.append(BASE_DIR)

from data_pipeline import input_fn

class Evaluator:
    def __init__(self, params, output_dir):
        self.dataset_dir = params.dataset_dir
        self.demo_set = params.demo_set
        self.output_dir = output_dir
        self.params = params
        self.models = {'disparity': DisparityNet(input_shape=(params.input_h, params.input_w, 3))}
        self.load_checkpoint(self.models['disparity'], os.path.join(output_dir, 'disparity_model'))

        # Datasets
        tf_records = [os.path.join(params.dataset_dir,file) for file in os.listdir(params.dataset_dir) if file.endswith('.tfrecords')]
        self.val_dataset = input_fn(tf_records[30:], load_option='val')
        self.images = []
        self.gt_depths = []
        for i, data in enumerate(self.val_dataset):
            image = data['images'][0][0]
            self.images.append(image)
            gt_depth = data['depths'].numpy()[0][0]
            self.gt_depths.append(gt_depth)

        print(f'Total Images: {len(self.images)}')

    def load_checkpoint(self, model, output_dir):
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=model)
        manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("No weights to Restores.")
            raise ValueError(f'No Weight to restore in {output_dir}')

    def do_demo(self, folder):
        scene = 'scene0001_00'
        save_dir = os.path.join(self.output_dir, 'predictions', scene)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        folder_dir = os.path.join(folder, scene, 'color', '*.jpg')
        images_files = sorted(glob.glob(folder_dir))
        print(f'doing demo on {os.path.dirname(folder_dir)}... ')
        print(f'saving prediction to {save_dir}...')
        for i, img_path in enumerate(images_files):
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.params.input_w, self.params.input_h))
            img_input = tf.expand_dims(tf.convert_to_tensor(img, tf.float32) / 255., 0)
            outputs = self.val_step(img_input)

            disp = np.squeeze(outputs['disparity0'].numpy())
            disp = visualize_colormap(disp)
            save_path = os.path.join(save_dir, f'{i}.png')

            big_image = np.zeros(shape=(self.params.input_h * 2, self.params.input_w, 3))
            big_image[:self.params.input_h, ...] = img
            big_image[self.params.input_h:, ...] = disp
            cv2.imwrite(save_path, cv2.cvtColor(big_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        print("\n-> Done!\n")

    def eval_depth(self):
        pred_depths = []
        pred_disps = []
        errors = []
        ratios = []

        # Predict
        print('doing evaluation...')
        for i, image in enumerate(self.images):
            img = tf.image.resize(image, [self.params.input_h, self.params.input_w])
            img_input = tf.expand_dims(img, axis=0)
            outputs = self.val_step(img_input)
            _, depth = disp_to_depth(outputs['disparity0'], min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
            depth *= 0.54

            pred_depths.append(depth.numpy())
            pred_disps.append(np.squeeze(outputs['disparity0'].numpy()))

        for i in range(len(pred_depths)):
            gt_depth = self.gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_depth = pred_depths[i][0]
            pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            # Median scaling
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            
            errors.append(compute_errors(gt_depth, pred_depth))

        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)
        np.savetxt(os.path.join(self.output_dir, 'mean_errors.txt'), mean_errors)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!\n")

    @tf.function
    def val_step(self, inputs):
        return self.models['disparity'](inputs, training=False)


if __name__ == '__main__':
    params = parser.parse_args()
    output_dir = os.path.join(PROJECT_DIR, 'results', params.identifier)

    c = Evaluator(params, output_dir)
    c.eval_depth()
    c.do_demo(params.demo_set)
    
