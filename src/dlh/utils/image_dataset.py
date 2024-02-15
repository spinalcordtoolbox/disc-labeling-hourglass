import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

from dlh.utils.train_utils import rand_crop_fn, rand_locked_fov_fn, transform_fn


class image_Dataset(Dataset):
    def __init__(self, images, targets=None, discs_labels=None, img_res=None, subjects_names=None, num_channel=None, use_flip=True, use_crop=False, use_lock_fov=False, load_mode='test'):  # initial logic happens like transform
        
        self.images = images
        self.targets = targets
        self.discs_labels = discs_labels
        self.img_res = img_res
        self.subjects_names = subjects_names
        self.num_channel = num_channel
        self.num_vis_joints = []
        self.use_flip = use_flip
        self.use_crop = use_crop
        self.use_lock_fov = use_lock_fov
        self.load_mode = load_mode

    def __len__(self):  # return count of sample we have
        return len(self.images)
    
    def get_posedata(self, msk, discs_list, num_ch=11):
        ys = msk.shape
        ys_ch = np.zeros([ys[0], ys[1], num_ch])

        if num_ch != 1:
            msk_uint = np.uint8(np.where(msk>0.2, 1, 0))
            num_labels, labels_im = cv2.connectedComponents(msk_uint)
            self.num_vis_joints.append(num_labels-1) # the <0> label is the background

            for i, num_disc in enumerate(discs_list):
                if num_disc <= num_ch:
                    num_label = i + 1  # label index cv2
                    y_i = msk * np.where(labels_im == num_label, 1, 0)
                    ys_ch[:,:, num_disc-1] = y_i
        
            vis = np.zeros((num_ch, 1))
            vis[discs_list[0]-1:discs_list[-1]] = 1
        else:
            ys_ch[:,:, 0] = msk
            vis = np.ones((num_ch, 1))
        return ys_ch, vis
    
    def rand_crop(self, image, mask, discs_labels, img_res, vis, min_discs=5, dy_disc=8, dx_disc=25):
        image, mask, included_discs, vis = rand_crop_fn(image, mask, discs_labels, img_res, vis, min_discs, dy_disc, dx_disc)
        return image, mask, included_discs, vis

    def rand_locked_fov(self, image, mask, discs_labels, img_res, vis, fov=(100,100)):
        out_image, out_mask, included_discs, vis = rand_locked_fov_fn(image, mask, discs_labels, img_res, vis, fov)
        return out_image, out_mask, included_discs, vis

    def transform(self, image, mask=None):
        if not mask is None:
            image, mask = transform_fn(image, mask, use_flip=self.use_flip)
            return image, mask
        else:
            image = transform_fn(image, mask, use_flip=self.use_flip)
            return image
    
    def __getitem__(self, index):
        
        image = self.images[index]
        if not self.targets is None:
            mask = self.targets[index]
            discs_labels = np.array(self.discs_labels[index])
            img_res = np.array(self.img_res[index])
            mask, vis  = self.get_posedata(mask, discs_labels[:,-1], num_ch=self.num_channel) # Split discs into different classes
            if self.use_crop:
                image, mask, discs_labels, vis = self.rand_crop(image, mask, discs_labels, img_res, vis, min_discs=6)
            if self.use_lock_fov:
                image, mask, discs_labels, vis = self.rand_locked_fov(image, mask, discs_labels, img_res, vis, fov=(150,150))
            t_image, t_mask = self.transform(image, mask)
            vis = torch.FloatTensor(vis)
        else:
            t_image = self.transform(image, mask=None)
            
        subject = self.subjects_names[index]

        if self.load_mode == 'train':
            return (t_image, t_mask, vis, subject)
        if self.load_mode == 'val':
            return (t_image, t_mask, vis)
        if self.load_mode == 'test':
            return (t_image, subject)

class image_Dataset2(Dataset):
    def __init__(self, images, targets, discs_labels=None, img_res=None, subjects_names=None, num_channel=None, use_flip=True, use_crop=False, use_lock_fov=False, load_mode='test'):  # initial logic happens like transform
        
        self.images = images
        self.targets = targets
        self.discs_labels = discs_labels
        self.img_res = img_res
        self.subjects_names = subjects_names
        self.num_channel = num_channel
        self.num_vis_joints = []
        self.use_flip = use_flip
        self.use_crop = use_crop
        self.use_lock_fov = use_lock_fov
        self.load_mode = load_mode

    def __len__(self):  # return count of sample we have
        return len(self.images)
    
    def get_posedata(self, msk, discs_list, num_ch=11):
        ys = msk.shape
        ys_ch = np.zeros([ys[0], ys[1], num_ch])

        if num_ch != 1:
            msk_uint = np.uint8(np.where(msk>0.2, 1, 0))
            num_labels, labels_im = cv2.connectedComponents(msk_uint)
            self.num_vis_joints.append(num_labels-1) # the <0> label is the background

            for i, num_disc in enumerate(discs_list):
                if num_disc <= num_ch:
                    num_label = i + 1  # label index cv2
                    y_i = msk * np.where(labels_im == num_label, 1, 0)
                    ys_ch[:,:, num_disc-1] = y_i
        
            vis = np.zeros((num_ch, 1))
            vis[discs_list[0]-1:discs_list[-1]] = 1
        else:
            ys_ch[:,:, 0] = msk
            vis = np.ones((num_ch, 1))
        return ys_ch, vis
    
    def rand_crop(self, image, mask, discs_labels, img_res, vis, min_discs=5, dy_disc=8, dx_disc=25):
        image, mask, included_discs, vis = rand_crop_fn(image, mask, discs_labels, img_res, vis, min_discs, dy_disc, dx_disc)
        return image, mask, included_discs, vis

    def rand_locked_fov(self, image, mask, discs_labels, img_res, vis, fov=(100,100)):
        out_image, out_mask, included_discs, vis = rand_locked_fov_fn(image, mask, discs_labels, img_res, vis, fov)
        return out_image, out_mask, included_discs, vis

    def transform(self, image, mask=None):
        if not mask is None:
            image, mask = transform2_fn(image, mask, use_flip=self.use_flip)
            return image, mask
        else:
            image = transform2_fn(image, mask, use_flip=self.use_flip)
            return image
    
    def __getitem__(self, index):
        
        image = self.images[index]
        mask = self.targets[index]
        discs_labels = np.array(self.discs_labels[index])
        img_res = np.array(self.img_res[index])
        mask, vis  = self.get_posedata(mask, discs_labels[:,-1], num_ch=self.num_channel) # Split discs into different classes
        if self.use_crop:
            image, mask, discs_labels, vis = self.rand_crop(image, mask, discs_labels, img_res, vis, min_discs=6)
        if self.use_lock_fov:
            image, mask, discs_labels, vis = self.rand_locked_fov(image, mask, discs_labels, img_res, vis, fov=(150,150))
        t_image, t_mask = self.transform(image, mask)
        vis = torch.FloatTensor(vis)
            
        subject = self.subjects_names[index]

        if self.load_mode == 'train':
            return (t_image, t_mask, vis, subject)
        if self.load_mode == 'val':
            return (t_image, t_mask, vis)
        if self.load_mode == 'test':
            return (t_image, subject)