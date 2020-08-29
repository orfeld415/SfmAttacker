import torch
from torch.autograd import Variable
from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
from models import PoseExpNet
from inverse_warp import pose_vec2mat
import torch.nn.functional as F

parser = argparse.ArgumentParser(
    description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str,
                    help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
parser.add_argument("--tracker-file", default=None, help="Object tracker numpy file")
# parser.add_argument("--perturbation", default=None, help="perturbations numpy file")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def resize2d(img, size):
    img = torch.tensor(img)
    return F.adaptive_avg_pool2d(img, size)


def getNetInput(sample):
    """ Returns model input given a sample of images snippet """
    args = parser.parse_args()
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


@torch.no_grad()
def main(perturbation, result_name):
    args = parser.parse_args()
    attack = False
    if args.tracker_file:
        attack = True
        loaded_perturbations = np.load(Path(perturbation))
        perturbations = [loaded_perturbations[i] for i in range(0, loaded_perturbations.shape[0])]
        print(loaded_perturbations.shape)
        noise_mask = np.load(Path(args.tracker_file))

    from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework

    weights = torch.load(args.pretrained_posenet, map_location=device)
    seq_length = int(weights['state_dict']['conv1.0.weight'].size(1) / 3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)

    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))
        ground_truth_array = np.zeros((len(framework), seq_length, 3, 4))

    for j, sample in enumerate(tqdm(framework)):
        tgt_img, ref_imgs = getNetInput(sample)
        imgs = sample['imgs']
        if attack:
            # Add noise to target image
            if j + 2 >= first_frame and j + 2 < last_frame:
                curr_mask = noise_mask[j - first_frame + 2].astype(np.int)
                w = curr_mask[2] - curr_mask[0]
                h = curr_mask[3] - curr_mask[1]
                noise_box = torch.tensor(
                    perturbations[j - first_frame][:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]])
                noise_box = noise_box.to(device)
                z_clamped = noise_box.clamp(-2, 2)
                z_clamped = z_clamped.to(device)
                #z_claped = noise_box.tanh()
                #tgt_img[0][:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] = z_claped
                tgt_img[0][:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] += z_clamped
                tgt_img[0] = tgt_img[0].clamp(-1, 1)

            # Add noise to reference images
            for k in range(5):
                ref_idx = k
                if k == 2:
                    # Skip target image
                    continue
                if k > 2:
                    # Since it is numbered: ref1, ref2, tgt, ref3, ref4
                    ref_idx = k - 1
                if j + k >= first_frame and j + k < last_frame:
                    curr_mask = noise_mask[j - first_frame + k].astype(np.int)
                    w = curr_mask[2] - curr_mask[0]
                    h = curr_mask[3] - curr_mask[1]
                    noise_box = torch.tensor(
                        perturbations[j - first_frame][:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]])
                    noise_box = noise_box.to(device)
                    # ref_imgs[ref_idx][0][:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] += noise_box
                    z_claped = noise_box.tanh()
                    ref_imgs[ref_idx][0][:, curr_mask[1]:curr_mask[3], curr_mask[0]:curr_mask[2]] = z_claped
                    # ref_imgs[ref_idx] = ref_imgs[ref_idx].clamp(-1, 1)

        _, poses = pose_net(tgt_img, ref_imgs)
        poses = poses.cpu()[0]
        poses = torch.cat([poses[:len(imgs) // 2], torch.zeros(1, 6).float(), poses[len(imgs) // 2:]])
        inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:, :3] @ transform_matrices
        final_poses[:, :, -1:] += first_inv_transform[:, -1:]

        if args.output_dir is not None:
            ground_truth_array[j] = sample['poses']
            predictions_array[j] = final_poses

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE', 'RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    if args.output_dir is not None:
        np.save(output_dir / 'ground_truth.npy', ground_truth_array)
        np.save(output_dir + result_name, predictions_array)


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1]) / np.sum(pred[:, :, -1] ** 2)
    ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE / snippet_length, RE / snippet_length


if __name__ == '__main__':
    first_frame = 691
    last_frame = 731
    main("noiseRight.npy", "predictions_perturbed_Right.npy")
    main("noiseLeft.npy", "predictions_perturbed_Left.npy")
