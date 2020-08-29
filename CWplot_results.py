import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import os
from scipy.misc import imresize
import argparse
from path import Path
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser(description='',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--tracker-file", default=None, help="Object tracker numpy file")
parser.add_argument("--perturbation", default=None, help="perturbations numpy file")
parser.add_argument("--ground-truth-results", default=None, help="ground truth results numpy file")
parser.add_argument("--perturbed-results", default=None, help="perturbed results numpy file")
parser.add_argument("--model-results", default=None, help="model results numpy file")
parser.add_argument("--animate", action='store_true')


def resize2d(img, size):
    img = torch.tensor(img)
    return F.adaptive_avg_pool2d(img, size)


def getAbsolutePoses(poses):
    for i in range(1, len(poses)):
        r = poses[i - 1, 1]
        poses[i] = r[:, :3] @ poses[i]
        poses[i, :, :, -1] = poses[i, :, :, -1] + r[:, -1]
    return poses


def getAbsoluteScale(poses, gt):
    for i in range(len(poses)):
        scale_factor = 1
        if np.sum(poses[i][:, :, -1] ** 2) != 0:
            scale_factor = np.sum(gt[i][:, :, -1] * poses[i][:, :, -1]) / np.sum(poses[i][:, :, -1] ** 2)
        poses[i][:, :, -1] = scale_factor * poses[i][:, :, -1]
    return poses


def animate(i):
    image = mpimg.imread(dataset_dir / '{}'.format(imgs[i]))
    image = imresize(image, (args.img_height, args.img_width)).astype(np.float32)
    image = image / 255
    image = (image * 2) - 1
    if i >= first_frame and i < last_frame:
        # Add the adversarial noise to image
        curr_mask = noise_mask[i - first_frame].astype(np.int)
        w = curr_mask[2] - curr_mask[0]
        h = curr_mask[3] - curr_mask[1]
        noise_box = torch.tensor(
            perturbations[i - first_frame][:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]])
        pert = np.transpose(noise_box, (1, 2, 0))  # .numpy()
        z_clamped = pert.clamp(-2, 2)
        # z_clamped = pert.tanh()
        image[curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] += z_clamped.numpy()  # pert / 2 + 1
        # image[curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] = z_clamped.numpy()
        image = np.clip(image, -1, 1)
    image = (image + 1) / 2

    # Update displayed image
    im.set_array(image)

    # Update trajectories
    data = np.hstack((xz_gt[0][:i, np.newaxis], xz_gt[1][:i, np.newaxis]))
    traj_gt.set_offsets(data)
    data = np.hstack((xz_pred[0][:i, np.newaxis], xz_pred[1][:i, np.newaxis]))
    traj_pred.set_offsets(data)
    data = np.hstack((xz_perturbed[0][:i, np.newaxis], xz_perturbed[1][:i, np.newaxis]))
    traj_perturbed.set_offsets(data)
    return traj_gt, traj_pred, traj_perturbed, im


if __name__ == '__main__':
    first_frame = 691
    last_frame = 731

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    loaded_perturbations = np.load(Path(args.perturbation))
    perturbations = [loaded_perturbations[i] for i in range(0, loaded_perturbations.shape[0])]
    perturbed = np.load(Path(args.perturbed_results))
    pred = np.load(Path(args.model_results))
    gt = np.load(Path(args.ground_truth_results))
    noise_mask = np.load(Path(args.tracker_file))

    perturbed = getAbsoluteScale(perturbed, gt)
    perturbed = getAbsolutePoses(perturbed)
    pred = getAbsoluteScale(pred, gt)
    pred = getAbsolutePoses(pred)
    gt = getAbsolutePoses(gt)

    # Get xz space from poses matrices
    xz_gt = (gt[:, 1][:, :, -1][:, 0], gt[:, 1][:, :, -1][:, 2])
    xz_pred = (pred[:, 1][:, :, -1][:, 0], pred[:, 1][:, :, -1][:, 2])
    xz_perturbed = (perturbed[:, 1][:, :, -1][:, 0], perturbed[:, 1][:, :, -1][:, 2])

    imgs = []
    for file in sorted(os.listdir(dataset_dir)):
        if file.endswith(".png"):
            imgs.append(file)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_aspect('equal')
    ax1.set_ylim(-200, 700)
    ax1.set_xlim(-500, 700)

    if (args.animate):
        traj_gt = ax1.scatter(0, 0, s=1, label='ground truth')
        traj_pred = ax1.scatter(0, 0, s=1, label='model prediction')
        traj_perturbed = ax1.scatter(0, 0, s=1, label='adversarial results')
        ax1.legend(handles=[traj_gt, traj_pred, traj_perturbed])

        img = mpimg.imread(dataset_dir / '000000.png')
        img = imresize(img, (args.img_height, args.img_width))
        im = ax2.imshow(img, animated=True)

        anim = animation.FuncAnimation(fig, animate, interval=100, blit=True)
        anim.save('animation.gif', writer='imagemagick')
    else:
        ax1.plot(xz_gt[0], xz_gt[1], label='ground truth')
        ax1.plot(xz_pred[0], xz_pred[1], label='model prediction')
        ax1.plot(xz_perturbed[0], xz_perturbed[1], label='adversarial results')
        ax1.legend()
    plt.show()
