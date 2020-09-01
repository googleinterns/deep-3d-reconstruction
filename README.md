# deep-3d-reconstruction

# Installation

Install Tensorflow v2.x and Anaconda libraries. The code was run in python 3.6 and GPU is expected but not required.
The codes are borrowed from [Here](https://github.com/ken012git/joint_point_based/tree/master/prepare_data/export_sens)

## Data

Download and pre-process the scannet data following instructions in the Scannet directory.

## Bounding Boxes

To get axis aligned bounding boxes for Scannet data, refer to this [repo](https://github.com/facebookresearch/votenet/tree/master/scannet). Project the boxes to the rgb images and annotate the heading angles for the boxes in the scenes. Follow guide on box_projection notebook to project and annotate the boxes.

## Training

Run `python data_pipeline.py` to create tensorflow examples for the data. Train the model by running `python train.py` from the pose_depth_estimation directory. After training, run `python eval.py` to evalute the model. The codes for training and evaluation codes were adapted from [Here](https://github.com/darylclimb/cvml_project/tree/master/depth/self_supervised_sfm)
