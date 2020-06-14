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

def resize2d(img, size):
    return (F.adaptive_avg_pool2d(img, size))

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
        #curr_mask = [50,50,90,90]#noise_mask[0].astype(np.int)
        noise = torch.randn((3,128,416))
        noise.requires_grad_(True)
        optimizer = torch.optim.Adam([noise], lr=2e-1)
        for epoch in range(20):
            CP = CompensatedPoses()
            rand_log_i = np.random.randint(39)
            for k in mask:
                sample = framework.getByIndex(k)
                i = k-mask[0]
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
                
                zeros = torch.tensor(0)
                m_ones = torch.tensor(-1)
                ones = torch.tensor(1)
                
                if i+2 < len(mask):
                    curr_mask = noise_mask[i+2].astype(np.int)
                    w = curr_mask[2]-curr_mask[0]
                    h = curr_mask[3]-curr_mask[1]
                    noise_box = resize2d(noise, (h,w))
                    z_clamped = noise_box.clamp(-1, 1)
                    tgt_img[0,:,curr_mask[1]:curr_mask[3],curr_mask[0]:curr_mask[2]] = z_clamped
                for j, ref in enumerate(ref_imgs):
                    if j == 1 and i+1 >= len(mask):
                        continue
                    if j == 2 and i+3 >= len(mask):
                        continue
                    if j == 3 and i+4 >= len(mask):
                        continue
                    if j == 0:
                        curr_mask = noise_mask[i+0].astype(np.int)
                    if j == 1:
                        curr_mask = noise_mask[i+1].astype(np.int)
                    if j == 2:
                        curr_mask = noise_mask[i+3].astype(np.int)
                    if j == 3:
                        curr_mask = noise_mask[i+4].astype(np.int)
                    w = curr_mask[2]-curr_mask[0]
                    h = curr_mask[3]-curr_mask[1]
                    #print('w: {}, h: {}'.format(w,h))
                    noise_box = resize2d(noise, (h,w))
                    z_clamped = noise_box.clamp(-1, 1)
                    ref[0,:,curr_mask[1]:curr_mask[3],curr_mask[0]:curr_mask[2]] = z_clamped


                _, pose = pose_exp_net(tgt_img, ref_imgs)
                pose_delta = CP.get(pose,save=True)
                optimizer.zero_grad()

                loss1 = self.criterion(pose_delta[:,0],zeros)
                loss2 = self.criterion(pose_delta[:,2],zeros)
                loss = loss1 + loss2
                if i == 33:
                    print("epoch {} loss: {}".format(epoch,loss.item()))
                    print("x: {}, z: {}".format(pose_delta[1,0],pose_delta[1,2]))

                loss.backward()

                optimizer.step()
    
                #plt.imshow(np.transpose(eta[0][0],(1,2,0)))
                #plt.show()
    
                #print(loss)
                #print("before x: {}, z: {}".format(pose1[1,0],pose1[1,2]))
                #print("after x: {}, z: {}".format(pose_delta[1,0],pose_delta[1,2]))
        noise.clamp(-1,1)
        return noise.detach().numpy()


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
pertubation = attacker.generate(framework, pose_net, mask)
np.save('/home/ziv/Desktop/sfm/SfmLearner-Pytorch/results/pertubations.npy', pertubation)
