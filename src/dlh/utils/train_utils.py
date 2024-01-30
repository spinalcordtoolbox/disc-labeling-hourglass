#===================================================
## Authors: 
# - Lucas Rouhier ()
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
#===================================================

import os
import numpy as np
import cv2
from scipy import signal
import copy
from random import randint
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from dlh.utils.transform_spe import RandomHorizontalFlip, ToTensor 
from dlh.utils.data2array import mask2label, get_midNifti

# normalize Image
def normalize(arr):
    ma = arr.max()
    mi = arr.min()
    return ((arr - mi) / (ma - mi))


# Useful function to generate a Gaussian Function on given coordinates. Used to generate groudtruth.
def label2MaskMap_GT(data, shape, res_im, c_dx=0, c_dy=0, radius=2, normalize=False):
    """
    Generate a Mask map from the coordenates
    :param shape: dimension of output
    :param data : input image
    :param radius: is the radius of the gaussian function
    :param normalize : bool for normalization.
    :return: a MxN normalized array
    """

    # Our 2-dimensional distribution will be over variables X and Y
    (M, N) = (shape[2], shape[1])
    if len(data) <= 2:
        # Output coordinates are reduced during post processing which poses a problem
        data = [0, data[0], data[1]]
    maskMap = []

    x, y = data[2], data[1]

    # Correct the labels
    x += c_dx
    y += c_dy

    X = np.linspace(0, M - 1, M)
    Y = np.linspace(0, N - 1, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Mean vector and covariance matrix
    mu = np.array([x, y])
    Sigma = np.array([[radius/res_im[1], 0], [0, radius/res_im[0]]]) # we divide by the resolution to have pixels

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # Normalization
    if normalize:
        Z *= (1 / np.max(Z))
    else:
        # 8bit image values (the loss go to inf+)
        Z *= (1 / np.max(Z))
        Z = np.asarray(Z * 255, dtype=np.uint8)

    maskMap.append(Z)

    if len(maskMap) == 1:
        maskMap = maskMap[0]

    return np.asarray(maskMap)


def extract_all(list_coord_label, res_im, shape_im):
    """
    Create groundtruth by creating gaussian Function for every ground truth points for a single image
    :param list_coord_label: list of ground truth coordinates
    :param res_im: image resolution
    :param shape_im: shape of output image with zero padding
    :return: a 2d heatmap image.
    """
    shape_tmp = (1, shape_im[0], shape_im[1])
    final = np.zeros(shape_tmp[1:])
    for coord in list_coord_label:
        train_lbs_tmp_mask = label2MaskMap_GT(coord, shape_tmp, res_im)
        np.maximum(final, train_lbs_tmp_mask, out=final)
    return np.expand_dims(final, axis=0)


class image_Dataset(Dataset):
    def __init__(self, images, targets=None, discs_labels=None, img_res=None, subjects_names=None, num_channel=None, use_flip=True, load_mode='test'):  # initial logic happens like transform
        
        self.images = images
        self.targets = targets
        self.discs_labels = discs_labels
        self.img_res = img_res
        self.subjects_names = subjects_names
        self.num_channel = num_channel
        self.num_vis_joints = []
        self.use_flip = use_flip
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
    
    def rand_crop(self, image, mask, discs_labels, img_res, min_discs=4, dy_disc=8, dx_disc=25):
        """
        Create a random crop for an image and its mask based on the number of visible discs.
        :param image: 2D image
        :param mask: 2D mask corresponding to the image
        :param discs_labels: Coordinates of the discs
        :param img_res: Image resolution (mm/pixel).
        :param min_discs: Minimum number of discs that have to be visible in the image.
        :param dy_disc: Vertical size of an intervertebral disc.
        :param dx_disc: Horizontal size of an intervertebral disc.
        """
        shape = image.shape
        if len(discs_labels) > min_discs:
            rand_num_discs = randint(min_discs, len(discs_labels)) # Get a random number of discs to keep
            rand_start_disc = randint(1, len(discs_labels)-rand_num_discs+1) # Get a random discs to start (the start disc is included)
        else:
            rand_num_discs = len(discs_labels)
            rand_start_disc = 1
        included_discs = discs_labels[rand_start_disc-1:rand_start_disc-1 + rand_num_discs]

        # Set not included discs masks to 0
        mask[:,:, ~np.in1d(np.arange(1, mask.shape[-1]+1), included_discs[:,-1])]=0

        # Get min and max discs coordinates
        disc_min_num = included_discs[0,-1]
        disc_max_num = included_discs[-1,-1]
        x_min_coord = min(included_discs[:,2]) # Smallest x coordinate for an included disc
        x_max_coord = max(included_discs[:,2]) # Biggest x coordinate for an included disc

        # Get coordinates of the first and the last included discs
        y_first_disc = included_discs[0,1]
        y_last_disc = included_discs[-1,1]
        x_first_disc = included_discs[0,2]
        x_last_disc = included_discs[-1,2]

        # Set vertical shift constraint with the last not included top (or bottom) discs or the maximum (or minimum) image size
        if disc_min_num != discs_labels[0,-1]:
            sup_max_shift = y_first_disc - discs_labels[np.where(discs_labels[:,-1]==disc_min_num-1)][:,1][0]
            sup_max_shift = round(sup_max_shift - (dy_disc/2)*img_res[0]) if round(sup_max_shift - (dy_disc/2)*img_res[0])>=0 else round(sup_max_shift) # Deal with disc thickness
        else:
            sup_max_shift = y_first_disc

        if disc_max_num != discs_labels[-1,-1]:
            inf_max_shift =  discs_labels[np.where(discs_labels[:,-1]==disc_max_num+1)][:,1][0] - y_last_disc
            inf_max_shift =  round(inf_max_shift - (dy_disc/2)*img_res[0]) if round(inf_max_shift - (dy_disc/2)*img_res[0])>=0 else round(inf_max_shift) # Deal with disc thickness
        else:
            inf_max_shift = shape[0] - y_last_disc

        sup_min_shift = round((dy_disc/2)*img_res[0]) if round((dy_disc/2)*img_res[0])<sup_max_shift else 0
        inf_min_shift = round((dy_disc/2)*img_res[0]) if round((dy_disc/2)*img_res[0])<inf_max_shift else 0

        # Set horizontal shift constraint based on the first disc coordinates
        left_max_shift = x_first_disc
        left_min_shift = round(x_first_disc - x_min_coord + (dx_disc/6)*img_res[1])
        right_max_shift = int(shape[1] - x_first_disc - 1)
        right_min_shift = round(x_max_coord - x_first_disc + dx_disc*img_res[1]) # Add disc horizontal width
        
        # Set random shifts based on the constraints
        x_shift_left = randint(left_min_shift, left_max_shift)
        x_shift_right = randint(right_min_shift, right_max_shift)
        y_shift_superior = randint(sup_min_shift, sup_max_shift)
        y_shift_inferior = randint(inf_min_shift, inf_max_shift)

        # Add random shift to the coodinates
        y_min = round(y_first_disc - y_shift_superior)
        y_max = round(y_last_disc + y_shift_inferior)
        x_min = round(x_first_disc - x_shift_left)
        x_max = round(x_first_disc + x_shift_right)

        return image[y_min:y_max,x_min:x_max], mask[y_min:y_max,x_min:x_max,:], included_discs

    def transform(self, image, mask=None):
        image = normalize(image)
        image = cv2.resize(image, (256, 256))
        image = np.expand_dims(image, -1)

        if not mask is None:
            resized_mask = np.zeros((256, 256, mask.shape[-1]))
            for i in range(mask.shape[-1]):
                resized_mask[:,:,i] = cv2.resize(mask[:, :, i], (256, 256))
            mask = resized_mask

        ## extract joints for pose model
        # Random horizontal flipping
        if self.use_flip:
            if not mask is None:
                image, mask = RandomHorizontalFlip()(pic=image, mask=mask)
            else:
                image = RandomHorizontalFlip()(pic=image)
        
        # Random vertical flipping
        # image,mask = RandomVerticalFlip()(image,mask)
        # random90 flipping
        temp_img = np.zeros((image.shape[0], image.shape[1], 3))
        temp_img[:,:,0:1]= image
        temp_img[:,:,1:2]= image
        temp_img[:,:,2:3]= image
        image = temp_img

        # Transform to tensor
        if not mask is None:
            image, mask = ToTensor()(pic=image, mask=mask)
            return image, mask
        else:
            image = ToTensor()(pic=image)
            return image
        
    
    def __getitem__(self, index):
        
        image = self.images[index]
        if not self.targets is None:
            mask = self.targets[index]
            discs_labels = np.array(self.discs_labels[index])
            img_res = np.array(self.img_res[index])
            mask, vis  = self.get_posedata(mask, discs_labels[:,-1], num_ch=self.num_channel) # Split discs into different classes
            image, mask, discs_labels = self.rand_crop(image, mask, discs_labels, img_res, min_discs=4)
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


def bluring2D(data, kernel_halfsize=3, sigma=1.0):
    x = np.arange(-kernel_halfsize,kernel_halfsize+1,1)
    y = np.arange(-kernel_halfsize,kernel_halfsize+1,1)
    xx, yy = np.meshgrid(x,y)
    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    filtered = signal.convolve(data, kernel, mode="same")
    return filtered

def rotate_img(img):
    img = np.rot90(img)
    img = np.flip(img, axis=1)
    return img


class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize


def save_epoch_res_as_image2(inputs, outputs, targets, out_folder, epoch_num, target_th=0.4, pretext=False, wandb_mode=False):
    max_epoch = 500
    target_th = target_th + (epoch_num/max_epoch*0.2)
    targets = targets.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    inputs = inputs.data.cpu().numpy()

    clr_vis_Y = []

    hues = np.linspace(0, 179, targets.shape[1], dtype=np.uint8)
    blank_ch = 255*np.ones_like(targets[0, 0], dtype=np.uint8)

    for Y in [targets, outputs]:
        for y, x in zip(Y, inputs):
            y_colored = np.zeros([y.shape[1], y.shape[2], 3], dtype=np.uint8)
            y_all = np.zeros([y.shape[1], y.shape[2]], dtype=np.uint8)
            for ych, hue_i in zip(y, hues):
                ych = ych/(np.max(np.max(ych))+0.00001)
                ych[np.where(ych<target_th)] = 0

                ych_hue = np.ones_like(ych, dtype=np.uint8)*hue_i
                ych = np.uint8(255*ych/(np.max(ych)+0.00001))
                
                colored_ych = np.zeros_like(y_colored, dtype=np.uint8)
                colored_ych[:, :, 0] = ych_hue
                colored_ych[:, :, 1] = blank_ch
                colored_ych[:, :, 2] = ych
                colored_y = cv2.cvtColor(colored_ych, cv2.COLOR_HSV2BGR)

                y_colored += colored_y
                y_all += ych

            x = np.moveaxis(x, 0, -1)
            x = x/np.max(x)*255

            x_3ch = np.zeros([x.shape[0], x.shape[1], 3])
            for i in range(3):
                x_3ch[:, :, i] = x[:, :, 0]
                        
            img_mix = np.uint8(x_3ch*0.5 + y_colored*0.5)
            clr_vis_Y.append(img_mix)
            
    targets, preds = np.concatenate(np.array(clr_vis_Y[:len(clr_vis_Y)//2]), axis=1), np.concatenate(np.array(clr_vis_Y[len(clr_vis_Y)//2:]), axis=1) 
    
    t = np.array(clr_vis_Y)
    t = np.transpose(t, [0, 3, 1, 2])
    trgts = make_grid(torch.Tensor(t), nrow=4)

    if pretext:
        txt = os.path.join(out_folder,f'/{epoch_num:0=4d}_test_result.png')
    else: 
        txt = os.path.join(out_folder,f'/epoch_{epoch_num:0=4d}_res2.png')
    res = np.transpose(trgts.numpy(), (1,2,0))
    
    if wandb_mode:
        return txt, res, targets, preds
    else:
        cv2.imwrite(txt, res)

##
def multivariate_gaussian(pos, mu, Sigma):
    """
    Return the multivariate Gaussian distribution on array.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N
    

class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []

def sigmoid(x):
    x = np.array(x)
    x = 1/(1+np.exp(-x))
    x[x<0.0] = 0
    return x

def save_attention(inputs, outputs, targets, att, target_th=0.5):
    targets = targets.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    inputs  = inputs.data.cpu().numpy()
    att     = att.detach().to('cpu')
    
    att = torch.sigmoid(att).numpy()
    att = np.uint8(att*255)
    att[att<128+80] = 0

    att     = cv2.resize(att, (256, 256))
    att     = cv2.applyColorMap(att, cv2.COLORMAP_JET)
    rgbatt  = copy.copy(inputs[0])
    rgbatt  = np.moveaxis(rgbatt, 0, -1)
    rgbatt = rgbatt*255*0.5+ att*0.5


    clr_vis_Y = []

    hues = np.linspace(0, 179, targets.shape[1], dtype=np.uint8)
    blank_ch = 255*np.ones_like(targets[0, 0], dtype=np.uint8)

    for Y in [targets, outputs]:
        for y, x in zip(Y, inputs):
            y_colored = np.zeros([y.shape[1], y.shape[2], 3], dtype=np.uint8)
            y_all = np.zeros([y.shape[1], y.shape[2]], dtype=np.uint8)
            for ych, hue_i in zip(y, hues):
                ych = ych/np.max(np.max(ych))
                ych[np.where(ych<target_th)] = 0

                ych_hue = np.ones_like(ych, dtype=np.uint8)*hue_i
                ych = np.uint8(255*ych/np.max(ych))
                
                colored_ych = np.zeros_like(y_colored, dtype=np.uint8)
                colored_ych[:, :, 0] = ych_hue
                colored_ych[:, :, 1] = blank_ch
                colored_ych[:, :, 2] = ych
                colored_y = cv2.cvtColor(colored_ych, cv2.COLOR_HSV2BGR)

                y_colored += colored_y
                y_all += ych

            x = np.moveaxis(x, 0, -1)
            x = x/np.max(x)*255

            x_3ch = np.zeros([x.shape[0], x.shape[1], 3])
            for i in range(3):
                x_3ch[:, :, i] = x[:, :, 0]
            
            img_mix = np.uint8(x_3ch*0.5 + y_colored*0.5)
            clr_vis_Y.append(img_mix)
            
    clr_vis_Y.append(rgbatt)
    t = np.array(clr_vis_Y)
    t = np.transpose(t, [0, 3, 1, 2])
    trgts = make_grid(torch.Tensor(t), nrow=4)
    txt = 'test/visualize/attention_visualization.png'
    res = np.transpose(trgts.numpy(), (1,2,0))
    cv2.imwrite(txt, res)

def loss_per_subject(pred, target, vis, vis_out, criterion):
    '''
    Return a list of loss corresponding to each image in the batch
    
    :param pred: Network prediction
    :param target: Ground truth mask
    '''
    losses = []
    if type(pred) == list:  # multiple output
        for p in pred:
            for idx in range(p.shape[0]):
                losses.append(criterion(p[idx], target[idx], vis[idx], vis_out[idx]).item())
    else:  # single output
        for idx in range(pred.shape[0]):
            losses.append(criterion(torch.unsqueeze(pred[idx], 0), torch.unsqueeze(target[idx], 0), torch.unsqueeze(vis[idx], 0), torch.unsqueeze(vis_out[idx], 0)).item())
    return losses

def apply_preprocessing(img_path, target_path='', num_channel=25):
    '''
    Load and apply preprocessing steps on input data
    :param img_path: Path to Niftii image
    :param target_path: Path to Niftii target mask
    '''
    image_in, res_image, shape_image = get_midNifti(img_path)
    image = (image_in - np.mean(image_in))/(np.std(image_in)+1e-100) # Equivalent to images_normalization function in dlh.utils.data2array
    image = normalize(image)
    image = image.astype(np.float32)
        
    if target_path != '':
        discs_labels, res_target, shape_target = mask2label(target_path)
        if res_image != res_target or shape_image != shape_target:
            raise ValueError(f'Image {img_path} and target {target_path} have different shapes or resolutions')
        discs_labels = [coord for coord in discs_labels if coord[-1] < num_channel+1] # Remove labels superior to the number of channels, especially 49 and 50 that correspond to the pontomedullary groove (49) and junction (50)
        mask = extract_all(discs_labels, res_image, shape_im=image_in.shape)
        mask = normalize(mask[0,:,:])
        mask = mask.astype(np.float32)
        return image, mask, discs_labels, res_image, image_in.shape
    else:
        return image, res_image, image_in.shape