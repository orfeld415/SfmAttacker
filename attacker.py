import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from models import PoseExpNet, DispNetS
from loss_functions import photometric_reconstruction_loss, explainability_loss, smooth_loss, compute_errors
from datasets.ziv_folders import SequenceFolder
from logger import TermLogger, AverageMeter
from tqdm import tqdm
from scipy.misc import imresize
from inverse_warp import pose_vec2mat
import matplotlib.pyplot as plt

class CompensatedPoses:
    def __init__(self):
        self.last = None

    def get(self, pose, save=False):
        poses = pose.cpu()[0]
        poses = torch.cat([poses[:5//2], torch.zeros(1,6).float(), poses[5//2:]])
        inv_transform_matrices = pose_vec2mat(poses, rotation_mode='euler').double()
        rot_matrices = torch.inverse(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]
        transform_matrices = torch.cat([rot_matrices, tr_vectors], axis=-1)
        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]

        if self.last is None:
            ret = final_poses
        else:
            r = self.last[1]
            ret = r[:,:3] @ final_poses
        if save:
            self.last = final_poses.detach()
        return ret[:,:,-1]

class Attacker:
    def __init__(self, clip_max=0.5, clip_min=-0.5):
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        pass

class FGSM(Attacker):
    """
    Fast Gradient Sign Method
    Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy.
    Explaining and Harnessing Adversarial Examples.
    ICLR, 2015
    """
    def __init__(self, eps=0.15, clip_max=0.5, clip_min=-0.5):
        super(FGSM, self).__init__(clip_max, clip_min)
        self.eps = eps
        self.criterion = torch.nn.L1Loss()

    def generate(self, framework, pose_exp_net, mask):
        perts = []
        CP = CompensatedPoses()
        for i in mask
            if i not in mask or i%5 !=1:
                continue
            imgs = sample['imgs']
            h,w,_ = imgs[0].shape
            if (h != 128 or w != 416):
                imgs = [imresize(img, (128, 416)).astype(np.float32) for img in imgs]
            
            imgs = [np.transpose(img, (2,0,1)) for img in imgs]
            ref_imgs = []
            for j, img in enumerate(imgs):
                img = torch.from_numpy(img).unsqueeze(0)
                img = ((img/255 - 0.5)/0.5).to(device)
                if j == len(imgs)//2:
                    tgt_img = img
                else:
                    ref_imgs.append(img)
            
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]

            tgt_img.requires_grad_()
            for ref in ref_imgs:
                ref.requires_grad_()
            
            zeros = torch.tensor(0)
            m_ones = torch.tensor(-1)
            ones = torch.tensor(1)
            _, pose = pose_exp_net(tgt_img, ref_imgs)
            pose_delta = CP.get(pose)
            loss = self.criterion(pose_delta[1,0],ones) + self.criterion(pose_delta[1,2],m_ones)
            pose_exp_net.zero_grad()
            #print(loss)
            pose1 = pose_delta
            eta = torch.zeros((5,1,3,128,416))
            for _ in range(50):
                refs = [ref_imgs[j] + eta[j+1] for j in range(4)]
                tgt_img0 = (tgt_img+eta[0]).clamp_(-1,1)
                refs = [refs[j].clamp_(-1,1) for j in range(4)]
                _, pose = pose_exp_net(tgt_img0, refs)

                pose_delta = CP.get(pose)
                loss1 = self.criterion(pose_delta[:,0],zeros)
                loss2 = self.criterion(pose_delta[:,2],zeros)
                loss = loss1 + loss2
                loss.backward()
                curr_mask = noise_mask[i-691+2].astype(np.int)
                eta[0,0,:,curr_mask[1]:curr_mask[3],curr_mask[0]:curr_mask[2]] -= 0.05 * torch.sign(tgt_img.grad.data)[0,:,curr_mask[1]:curr_mask[3],curr_mask[0]:curr_mask[2]]
                for j, ref in enumerate(ref_imgs):
                    if j == 0:
                        curr_mask = noise_mask[i-691+0].astype(np.int)
                    if j == 1:
                        curr_mask = noise_mask[i-691+1].astype(np.int)
                    if j == 2:
                        curr_mask = noise_mask[i-691+3].astype(np.int)
                    if j == 3:
                        curr_mask = noise_mask[i-691+4].astype(np.int)
                    eta[j+1,0,:,curr_mask[1]:curr_mask[3],curr_mask[0]:curr_mask[2]] -= 0.05 * torch.sign(ref_imgs[j].grad.data)[0,:,curr_mask[1]:curr_mask[3],curr_mask[0]:curr_mask[2]]
                eta.clamp_(-1, 1)
                pose_exp_net.zero_grad()

            #plt.imshow(np.transpose(eta[0][0],(1,2,0)))
            #plt.show()
            bla1 = tgt_img + eta[0]
            bla2 = [ref_imgs[j] + eta[j+1] for j in range(4)]
            _, pose = pose_exp_net(bla1, bla2)

            pose_delta = CP.get(pose,save=True)
            loss = self.criterion(pose_delta[1,0],ones) + self.criterion(pose_delta[1,2],m_ones)
            #print(loss)
            print("before x: {}, z: {}".format(pose1[1,0],pose1[1,2]))
            print("after x: {}, z: {}".format(pose_delta[1,0],pose_delta[1,2]))
            perts.append(eta[1].detach().numpy())
            perts.append(eta[2].detach().numpy())
            perts.append(eta[0].detach().numpy())
            perts.append(eta[3].detach().numpy())
            perts.append(eta[4].detach().numpy())
        return perts


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
weights = torch.load('/home/ziv/Desktop/sfm/SfmLearner-Pytorch/models/exp_pose_model_best.pth.tar', map_location=device)
seq_length = int(weights['state_dict']['conv1.0.weight'].size(1)/3)
pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
pose_net.load_state_dict(weights['state_dict'], strict=False)

from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework
framework = test_framework('/home/ziv/Desktop/sfm/dataset', ['09'], 5)

attacker = FGSM()
mask = range(691,730)
noise_mask = np.load('/home/ziv/Desktop/sfm/SfmLearner-Pytorch/results/tracker_out.npy')
pertubations = np.array(attacker.generate(framework, pose_net, mask))
np.save('/home/ziv/Desktop/sfm/SfmLearner-Pytorch/results/pertubations.npy', pertubations)
