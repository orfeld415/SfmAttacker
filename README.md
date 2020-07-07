# Visual Odometry Adversarial Attacker

The project is an adversarial attacker on [SfMLearner model](https://github.com/ClementPinard/SfmLearner-Pytorch) pose predictor and is mainly built on SfMLearner codebase.
SfmLearner implements the system described in the paper:
[Unsupervised Learning of Depth and Ego-Motion from Video](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/)

SfMLearner uses a PoseNet to predict the object trajectory. In this project we try to attack PoseNet model by adding adversarial noise on objects thus increasing the model error:

![Alt text](misc/Figure_1.png?raw=true)

### Prerequisite

```
pip3 install -r requirements.txt
```

### Preparing data

This attacker currently aims KITTI visual odometry dataset. Some adjustments need to be made before applying this framework on different datasets.

To use KITTI dataset, download the colored dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

### Object tracker

This project contains a very basic object tracker obj_tracker.py
It allows you to track a rectangular object for n frames and outputs a nx4 matrix where the i-th row corresponds with the i-th frame you tracked and has the following format:

```
left_x up_y right_x down_y
```

An example use:

```
python obj_tracker.py --dataset-dir [PATH_TO_DATASET]/sequences/09/image_2/ -o tracked_obj.npy
```

### Attacker

The attacker first initiallizes a random noise to add for each attacked frame. It needs a tracked object matrix with the same format given by the object tracker in order to apply the correct transformation on the noise. Then it tries to adjust the noise to maximize the distance between the model's original results and the attacked results.

An example use:

```
python attacker.py models/exp_pose_model_best.pth.tar --tracker-file tracked_obj.npy --dataset-dir [PATH_TO_DATASET]/ -o noise.npy
```

### Get model's result

After retrieving a noise image from attacker.py we can get the model's results on the adversarial video with the following command

```
python test_pose.py models/exp_pose_model_best.pth.tar --dataset-dir [PATH_TO_DATASET] --sequences 09 --output-dir results/ --tracker-file tracker_out.npy --perturbation noise.npy
```

You can also get the original results by removing the flags --tracker-file and --perturbation

### Plot results

To visualize the results we use plot_results.py, which can plot the trajectory and show the adversarial video created.
To plot:

```
python plot_results.py --dataset-dir [PATH_TO_DATASET]/sequences/09/image_2/ --tracker-file tracker_out.npy --perturbation noise.npy --ground-truth-results results/ground_truth.npy --perturbed-results results/predictions_perturbed.npy --model-results results/predictions.npy
```

To also show the adversarial video add an --animte flag:

```
python plot_results.py --dataset-dir [PATH_TO_DATASET]/sequences/09/image_2/ --tracker-file tracker_out.npy --perturbation noise.npy --ground-truth-results results/ground_truth.npy --perturbed-results results/predictions_perturbed.npy --model-results results/predictions.npy --animate
```
