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

To use KITTI dataset, download the "colored dataset" and the "odometry ground truth" from [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

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

Inside obj_tracker.py you can set first_frame and last_frame to select which part of the video should be attacked. this is relevant to all the following files.

### Attacker

The attacker first initiallizes a random noise image to add for each attacked frame. It needs a tracked object matrix with the same format given by the object tracker in order to apply the correct transformation on the noise. Then it tries to adjust the noise to maximize the distance between the model's original results and the attacked results.

An example use:

```
python CWl2attacker.py models/exp_pose_model_best.pth.tar --tracker-file L_tracked_obj.npy --dataset-dir [PATH_TO_DATASET]
```

### Plot results

To visualize the results we use plot_results.py, which can plot the trajectory and show the adversarial video created.
To plot:

```
python CWplot_results.py --dataset-dir [PATH_TO_DATASET]\sequences\09\image_2\ --tracker-file L_tracked_obj.npy --perturbation noiseRight.npy --ground-truth-results resultsL/ground_truth.npy --perturbed-results results/predictions_perturbed_Right.npy --model-results results/predictions_perturbed.npy
```

To also show the adversarial video add an --animte flag:

```
python CWplot_results.py --dataset-dir [PATH_TO_DATASET]\sequences\09\image_2\ --tracker-file L_tracked_obj.npy --perturbation noiseRight.npy --ground-truth-results resultsL/ground_truth.npy --perturbed-results results/predictions_perturbed_Right.npy --model-results results/predictions_perturbed.npy --animte
```

### Reproduce the results

All hyperparameters are already set within the files.
The value of c is set to 360. You can change it to reproduce all the experiments.
The results were produced on KITTI's visual odometry sequence 09.

To reproduce the results run the following commands: 

1. To attack with the original attacker run:

```
python attacker2.py models/exp_pose_model_best.pth.tar --tracker-file L_tracked_obj.npy --dataset-dir [PATH_TO_DATASET]  -o attackernoise.npy
```

2. To plot the original attacker results run:

```
python plot_results.py --dataset-dir [PATH_TO_DATASET]/sequences/09/image_2/ --tracker-file tracked_obj.npy --perturbation attackernoise.npy --ground-truth-results results/ground_truth.npy --perturbed-results results/predictions_perturbed_attacker.npy --model-results results/predictions_perturbed.npy --animate
```

3.To run the new attack (with both right and left turns) run:

```
python CWl2attacker.py models/exp_pose_model_best.pth.tar --tracker-file L_tracked_obj.npy --dataset-dir [PATH_TO_DATASET]/dataset/
```

4.To plot the new attack results run (left attack): 

```
python CWplot_results.py --dataset-dir [PATH_TO_DATASET]\sequences\09\image_2\ --tracker-file L_tracked_obj.npy --perturbation noiseLeft.npy --ground-truth-results results/ground_truth.npy --perturbed-results results/predictions_perturbed_Left.npy --model-results results/predictions_perturbed.npy --animate
```

5.To plot the new attack results run (right attack): 

```
python CWplot_results.py --dataset-dir [PATH_TO_DATASET]\sequences\09\image_2\ --tracker-file L_tracked_obj.npy --perturbation noiseRight.npy --ground-truth-results results/ground_truth.npy --perturbed-results results/predictions_perturbed_Right.npy --model-results results/predictions_perturbed.npy --animate
```

