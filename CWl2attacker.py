import numpy as np
import torch
import torch.utils.data
from path import Path
from models import PoseExpNet
from tqdm import tqdm
from inverse_warp import pose_vec2mat
import argparse

from scipy.misc import imresize

parser = argparse.ArgumentParser(description='Script for attacking sfm model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_weights", type=str, help="pretrained PoseNet weights path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
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

    for i in range(1, len(poses)):
        r = poses[i - 1][1]
        poses[i] = r[:, :3] @ poses[i]
        poses[i][:, :, -1] = poses[i][:, :, -1] + r[:, -1]
    return poses[-1][:, :, -1]


def getNetInput(sample):
    """ Returns model input given a sample of images snippet """
    imgs = sample['imgs']
    h, w, _ = imgs[0].shape
    if h != args.img_height or w != args.img_width:
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


class Attacker:
    def __init__(self, framework, pose_net, adv_framework, look_ahead=60):
        self.look_ahead = look_ahead  # Number of frames to look ahead when calculating loss
        self.pose_net = pose_net
        self.framework = framework
        self.criterion = torch.nn.MSELoss()
        self.adv_framework = adv_framework

    def generate(self, noise_mask, first_frame, last_frame, first_adv_frame, best=None):
        """ Generates an adversarial example """

        # Initialize random noise from uniform distribution between (-1, 1)
        noises = []
        for i in range(first_frame, last_frame):
            curr_mask = noise_mask[i - first_frame].astype(np.int)
            w = curr_mask[2] - curr_mask[0]
            h = curr_mask[3] - curr_mask[1]
            noises += [(torch.rand((3, h, w)) * 2 - 1) * 1]

        if best is not None:
            best = torch.tensor(best)
            rs = [noise.norm().item() for noise in best]
            nrs = [(torch.rand(1).item() * r) for r in rs]
            noises = [(noise / noise.norm().item()) * nrs[i] for i, noise in enumerate(noises)]

        for noise in noises:
            noise.requires_grad_(True)

        optimizer = torch.optim.Adam(noises, lr=1e-1)

        # Get model adverserial "class" results (without attack)
        poses = []
        for i in tqdm(range(first_adv_frame, first_adv_frame + (last_frame - first_frame))):
            sample = self.adv_framework.getByIndex(i)
            tgt_img, ref_imgs = getNetInput(sample)
            _, pose = self.pose_net(tgt_img, ref_imgs)
            poses.append(pose)
        adv_poses = poses

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
        criterions = [(torch.nn.MSELoss(), torch.nn.MSELoss()) for _ in range(first_frame, last_frame)]
        final_loss = 0
        for epoch in range(50):
            poses = []
            for k in tqdm(range(first_frame, last_frame + self.look_ahead)):
                sample = self.framework.getByIndex(k)
                i = k - first_frame
                tgt_img, ref_imgs = getNetInput(sample)
                if k + 2 < last_frame:
                    curr_mask = noise_mask[i + 2].astype(np.int)
                    noise_box = noises[k - first_frame + 2]
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
                    idx = j
                    if j >= 2:
                        idx += 1
                    noise_box = noises[k - first_frame + idx]
                    z_clamped = noise_box.clamp(-2, 2)
                    z_clamped = z_clamped.to(device)
                    ref[0, :, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] += z_clamped
                    ref_imgs[j] = ref.clamp(-1, 1)

                _, pose = self.pose_net(tgt_img, ref_imgs)
                poses.append(pose)

            # Get attacked absoulte poses
            absolute_attacked_pose = getAbsolutePoses(poses)

            c = 360
            loss = 0
            for i in range(first_frame, last_frame):
                curr_mask = noise_mask[i - first_frame].astype(np.int)
                loss += c * criterions[i - first_frame][0](adv_poses[i - first_frame], poses[i - first_frame]) + \
                        criterions[i - first_frame][1](noises[i - first_frame],
                                                       torch.zeros_like(noises[i - first_frame])).to(device)
            total_loss = loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            final_loss = self.criterion(absolute_attacked_pose, orig_results)
            print("epoch {} loss: {}".format(epoch, loss.item()))
            print("\u0394x: {}, \u0394z: {}".format(absolute_attacked_pose[1, 0] - orig_results[1, 0],
                                                    absolute_attacked_pose[1, 2] - orig_results[1, 2]))

        # Clamp final noise to the range (-1,1) same as the net recieves
        # noises = [noise.clamp(-1, 1) for noise in noises]
        # noises = [noise.detach().numpy() for noise in noises]
        # for i, noise in enumerate(noises):
        #     z = np.zeros((3, args.img_height, args.img_width))
        #     curr_mask = noise_mask[i].astype(np.int)
        #     z[:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] = noise
        #     noises[i] = z

        # ~~~~~
        weights = torch.load("models/exp_pose_model_best.pth.tar", map_location=device)
        seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
        predictions_array = np.zeros((len(self.framework), seq_length, 3, 4))

        for k, sample in enumerate(tqdm(self.framework)):
            imgs = sample['imgs']
            i = k - first_frame
            tgt_img, ref_imgs = getNetInput(sample)

            if k >= first_frame:
                if k + 2 < last_frame:
                    curr_mask = noise_mask[i + 2].astype(np.int)
                    noise_box = noises[k - first_frame + 2]
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
                    idx = j
                    if j >= 2:
                        idx += 1
                    noise_box = noises[k - first_frame + idx]
                    z_clamped = noise_box.clamp(-2, 2)
                    z_clamped = z_clamped.to(device)
                    ref[0, :, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] += z_clamped
                    ref_imgs[j] = ref.clamp(-1, 1)

            _, poses = self.pose_net(tgt_img, ref_imgs)
            poses = poses.cpu()[0]
            poses = torch.cat([poses[:len(imgs) // 2], torch.zeros(1, 6).float(), poses[len(imgs) // 2:]])
            inv_transform_matrices = pose_vec2mat(poses, rotation_mode='euler').detach().numpy().astype(np.float64)

            rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
            tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]

            transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

            first_inv_transform = inv_transform_matrices[0]
            final_poses = first_inv_transform[:, :3] @ transform_matrices
            final_poses[:, :, -1:] += first_inv_transform[:, -1:]

            predictions_array[k] = final_poses

        noises = [noise.detach().numpy() for noise in noises]
        for i, noise in enumerate(noises):
            z = np.zeros((3, args.img_height, args.img_width))
            curr_mask = noise_mask[i].astype(np.int)
            z[:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] = noise
            noises[i] = z
        # ~~~~~~

        return total_loss.item(), np.stack(noises, axis=0), final_loss, predictions_array
        # return total_loss.item(), noises, final_loss


def main():
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
    loss, pertubationLeft, final_loss, predictions_array = attacker.generate(noise_mask, 691, 731, 187)
    for _ in range(4):
        tmploss, tmppertubationLeft, tmp_final_loss, predictions_array_tmp = attacker.generate(noise_mask, 691, 731,
                                                                                               187, pertubationLeft)
        if tmploss < loss:
            loss = tmploss
            pertubationLeft = tmppertubationLeft
            final_loss = tmp_final_loss
            predictions_array = predictions_array_tmp
    print('left loss: ' + str(final_loss))
    np.save("noiseLeft.npy", pertubationLeft)
    np.save("results/predictions_perturbed_Left.npy", predictions_array)

    loss, pertubationRight, final_loss, predictions_array = attacker.generate(noise_mask, 691, 731, 90)
    for _ in range(4):
        tmploss, tmppertubationRight, tmp_final_loss, predictions_array_tmp = attacker.generate(noise_mask, 691, 731,
                                                                                                90,
                                                                                                pertubationRight)
        if tmploss < loss:
            loss = tmploss
            pertubationRight = tmppertubationRight
            final_loss = tmp_final_loss
            predictions_array = predictions_array_tmp

    print('right loss: ' + str(final_loss))
    np.save("noiseRight.npy", pertubationRight)
    np.save("results/predictions_perturbed_Right.npy", predictions_array)


    print('Attacked!')


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parser.parse_args()
    main()
