import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import os
from scipy.misc import imresize
import argparse
from path import Path

parser = argparse.ArgumentParser(description='',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--animate", action='store_true')

def getAbsolutePoses(poses):
    for i in range(1,len(poses)):
        r = poses[i-1,1]
        poses[i] = r[:,:3] @ poses[i]
        poses[i,:,:,-1] = poses[i,:,:,-1] + r[:,-1]
    return poses

def getAbsoluteScale(poses, gt):
    for i in range(len(poses)):
        scale_factor = 1
        if np.sum(poses[i][:,:,-1] ** 2) != 0:
            scale_factor = np.sum(gt[i][:,:,-1] * poses[i][:,:,-1])/np.sum(poses[i][:,:,-1] ** 2)
        poses[i][:,:,-1] = scale_factor * poses[i][:,:,-1]
    return poses

def animate(i, traj_gt, traj_pred, traj_pertubated, im, xz_gt, xz_pred, xz_pertubated, dataset_dir, imgs):
    i = i+600
    image = mpimg.imread(dataset_dir/'{}'.format(imgs[i]))
    image = imresize(image, (128, 416)).astype(np.float32)
    image = image/255
    if i>= 691 and i<= 730:
        pert = pertubations[i-691][0]
        pert = np.transpose(pert, (1,2,0))
        image = image+(pert)/2
        image = np.clip(image,0,1)

        im.set_array(image)
    else:
        im.set_array(image)
    data = np.hstack((xz_gt[0][:i,np.newaxis], xz_gt[1][:i, np.newaxis]))
    traj_gt.set_offsets(data)
    data = np.hstack((xz_pred[0][:i,np.newaxis], xz_pred[1][:i, np.newaxis]))
    traj_pred.set_offsets(data)
    data = np.hstack((xz_pertubated[0][:i,np.newaxis], xz_pertubated[1][:i, np.newaxis]))
    traj_pertubated.set_offsets(data)
    return traj_gt, traj_pred, traj_pertubated, im

def main():
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    pertubated = np.load('results/predictions_pertubated.npy')
    pred = np.load('results/predictions.npy')
    gt = np.load('results/ground_truth.npy')

    pertubated = getAbsoluteScale(pertubated, gt)
    pertubated = getAbsolutePoses(pertubated)
    pred = getAbsoluteScale(pred, gt)
    pred = getAbsolutePoses(pred)
    gt = getAbsolutePoses(gt)
    xz_gt = (gt[:,1][:,:,-1][:,0], gt[:,1][:,:,-1][:,2])
    xz_pred = (pred[:,1][:,:,-1][:,0], pred[:,1][:,:,-1][:,2])
    xz_pertubated = (pertubated[:,1][:,:,-1][:,0], pertubated[:,1][:,:,-1][:,2])

    imgs = []
    for file in sorted(os.listdir(dataset_dir)):
        if file.endswith(".png"):
            imgs.append(file)

    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_aspect('equal')
    ax1.set_ylim(-200,600)
    ax1.set_xlim(-500,700)

    if(args.animate):
        traj_gt = ax1.scatter(0,0,s=1, label='ground truth')
        traj_pred = ax1.scatter(0,0,s=1, label='model prediction')
        traj_pertubated = ax1.scatter(0,0,s=1, label='perturbed')
        ax1.legend(handles=[traj_gt,traj_pred,traj_pertubated])
    
        img = mpimg.imread(dataset_dir/'000000.png')
        img = imresize(img, (args.img_height, args.img_width))
        im = ax2.imshow(img, animated = True)

        animation.FuncAnimation(fig, animate, interval=100, blit=True,
                                    fargs=(traj_gt,traj_pred, traj_pertubated, im,
                                    xz_gt, xz_pred, xz_pertubated, dataset_dir, imgs))
    else:
        ax1.plot(xz_gt[0], xz_gt[1], label='ground truth')
        ax1.plot(xz_pred[0], xz_pred[1], label='model prediction')
        ax1.plot(xz_pertubated[0], xz_pertubated[1], label='perturbed')
        ax1.legend()
    plt.show()

if __name__ == '__main__':
    pertubations = np.load('/home/ziv/Desktop/sfm/SfmLearner-Pytorch/results/pertubations.npy')
    main()
