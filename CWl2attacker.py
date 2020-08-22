import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from path import Path
from models import PoseExpNet
from tqdm import tqdm
from scipy.misc import imresize
from inverse_warp import pose_vec2mat
import argparse

parser = argparse.ArgumentParser(description='Script for attacking sfm model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_weights", type=str, help="pretrained PoseNet weights path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
#parser.add_argument("--output-file", "-o", default=None, help="Output numpy file")
parser.add_argument("--tracker-file", default=None, help="Object tracker numpy file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")


def getAbsolutePoses(poses):
    """ Return absolute poses from poses snippets (relative poses) """
    poses = np.array(poses)

    for i, pose in enumerate(poses):
        pose = pose.cpu()[0]
        pose = torch.cat([pose[:5 // 2], torch.zeros(1, 6).float(), pose[5 // 2:]])
        inv_transform_matrices = pose_vec2mat(pose, rotation_mode='euler').double()
        rot_matrices = torch.inverse(inv_transform_matrices[:, :, :3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]
        transform_matrices = torch.cat([rot_matrices, tr_vectors], axis=-1)
        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:, :3] @ transform_matrices
        final_poses[:, :, -1:] += first_inv_transform[:, -1:]
        poses[i] = final_poses
        # print(final_poses)

    for i in range(1, len(poses)):
        r = poses[i - 1][1]
        poses[i] = r[:, :3] @ poses[i]
        poses[i][:, :, -1] = poses[i][:, :, -1] + r[:, -1]

    return poses[-1][:, :, -1]


def resize2d(img, size):
    return (F.adaptive_avg_pool2d(img, size))


def getNetInput(sample):
    """ Returns model input given a sample of images snippet """
    imgs = sample['imgs']
    h, w, _ = imgs[0].shape
    if (h != args.img_height or w != args.img_width):
        imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]

    imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]
    ref_imgs = []
    for j, img in enumerate(imgs):
        img = torch.from_numpy(img).unsqueeze(0)
        img = ((img / 255 - 0.5) / 0.5).to(device)
        if j == len(imgs) // 2:
            tgt_img = img
        else:
            ref_imgs.append(img)

    tgt_img = tgt_img.to(device)
    ref_imgs = [img.to(device) for img in ref_imgs]

    return tgt_img, ref_imgs


class Attacker():
    def __init__(self, framework, pose_net, adv_framework, look_ahead=60):
        self.look_ahead = look_ahead  # Number of frames to look ahead when calculating loss
        self.pose_net = pose_net
        self.framework = framework
        self.criterion = torch.nn.MSELoss()
        self.adv_framework = adv_framework

    def generate(self, noise_mask, first_frame, last_frame, first_adv_frame):
        """ Generates an adversarial example """
        num_frames = last_frame - first_frame + 1

        # Initialize random noise from uniform distribution between (-1, 1)
        noises = [(torch.rand((3, args.img_height, args.img_width)) * 2 - 1) * 1 for _ in
                  range(first_frame, last_frame)]
        print(noises[0].shape)
        for noise in noises:
            noise.requires_grad_(True)

        # Initialize optimizer
        optimizers = []
        for noise in noises:
            optimizers.append(torch.optim.Adam([noise], lr=1e-1))

        # Get model adverserial "class" results (without attack)
        poses = []

        for i in tqdm(range(first_adv_frame, first_adv_frame + (last_frame - first_frame))):
            sample = self.adv_framework.getByIndex(i)
            tgt_img, ref_imgs = getNetInput(sample)
            _, pose = self.pose_net(tgt_img, ref_imgs)
            poses.append(pose)
        adv_poses = poses
        # orig_results = getAbsolutePoses(poses).detach().numpy()
        # orig_results = torch.from_numpy(orig_results).double()

        # Get model original results (without attack)
        poses = []
        for i in tqdm(range(first_frame, last_frame + self.look_ahead)):
            sample = self.framework.getByIndex(i)
            tgt_img, ref_imgs = getNetInput(sample)
            _, pose = self.pose_net(tgt_img, ref_imgs)
            poses.append(pose)
        orig_results = getAbsolutePoses(poses).detach().numpy()
        orig_results = torch.from_numpy(orig_results).double()

        # Train adversarial example
        criterions = [torch.nn.MSELoss() for _ in range(first_frame, last_frame)]
        for epoch in range(250):
            poses = []
            for k in tqdm(range(first_frame, last_frame + self.look_ahead)):
                sample = self.framework.getByIndex(k)
                i = k - first_frame
                tgt_img, ref_imgs = getNetInput(sample)

                # Add noise to target frame
                if k + 2 < last_frame:
                    curr_mask = noise_mask[i + 2].astype(np.int)
                    w = curr_mask[2] - curr_mask[0]
                    h = curr_mask[3] - curr_mask[1]
                    noise_box = noises[k - first_frame][:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]]
                    z_clamped = noise_box.clamp(-2, 2)
                    z_clamped = z_clamped.to(device)
                    tgt_img[0, :, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] += z_clamped
                    tgt_img = tgt_img.clamp(-1, 1)

                # Add noises to reference frames
                for j, ref in enumerate(ref_imgs):
                    if j == 0 and k >= last_frame:
                        continue
                    if j == 1 and k + 1 >= last_frame:
                        continue
                    if j == 2 and k + 3 >= last_frame:
                        continue
                    if j == 3 and k + 4 >= last_frame:
                        continue
                    if j == 0:
                        curr_mask = noise_mask[i + 0].astype(np.int)
                    if j == 1:
                        curr_mask = noise_mask[i + 1].astype(np.int)
                    if j == 2:
                        curr_mask = noise_mask[i + 3].astype(np.int)
                    if j == 3:
                        curr_mask = noise_mask[i + 4].astype(np.int)
                    w = curr_mask[2] - curr_mask[0]
                    h = curr_mask[3] - curr_mask[1]
                    noise_box = noises[k - first_frame][:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]]
                    z_clamped = noise_box.clamp(-2, 2)
                    z_clamped = z_clamped.to(device)
                    ref[0, :, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] += z_clamped
                    ref = ref.clamp(-1, 1)

                _, pose = self.pose_net(tgt_img, ref_imgs)
                poses.append(pose)

            # Get attacked absoulte poses
            absolute_attacked_pose = getAbsolutePoses(poses)

            ##### Minimize the negative loss <-> maximize the loss
            for i in range(first_frame, last_frame):
                loss = criterions[i - first_frame](adv_poses[i - first_frame], poses[i - first_frame])
                optimizers[i - first_frame].zero_grad()
                loss.backward(retain_graph=True)
                optimizers[i - first_frame].step()

            loss = -1 * self.criterion(absolute_attacked_pose, orig_results)
            print("epoch {} loss: {}".format(epoch, loss.item()))
            print("\u0394x: {}, \u0394z: {}".format(absolute_attacked_pose[1, 0] - orig_results[1, 0],
                                                    absolute_attacked_pose[1, 2] - orig_results[1, 2]))

        # Clamp final noise to the range (-1,1) same as the net recieves
        noises = [noise.clamp(-1, 1) for noise in noises]
        noises = [noise.detach().numpy() for noise in noises]
        print(noises[0].shape)
        return np.stack(noises, axis=0)


def main():
    #output_file = Path(args.output_file)
    tracker_file = Path(args.tracker_file)
    pretrained_weights = Path(args.pretrained_weights)
    dataset_dir = Path(args.dataset_dir)

    # Load pretrained model 
    weights = torch.load(pretrained_weights, map_location=device)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    # Set Kitti framework for sequence number 09 with 5-snippet samples.
    from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework
    framework = test_framework(dataset_dir, ['09'], 5)
    adv_framework = test_framework(dataset_dir, ['00'], 5)

    attacker = Attacker(framework, pose_net, adv_framework)
    noise_mask = np.load(tracker_file)
    pertubationLeft = attacker.generate(noise_mask, 691, 731, 187)
    pertubationRight = attacker.generate(noise_mask, 691, 731, 90)
    np.save("noiseLeft.npy", pertubationLeft)
    np.save("noiseRight.npy", pertubationRight)

    print('Attacked!')


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parser.parse_args()
    main()
