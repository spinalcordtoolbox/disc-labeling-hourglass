#===================================================
## Authors: 
# - Lucas Rouhier ()
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
#===================================================

import os
import cv2
import numpy as np
import torch
from progress.bar import Bar
from sklearn.utils.extmath import cartesian
from torchvision.utils import make_grid

from dlh.utils.train_utils import apply_preprocessing
from dlh.utils.data2array import get_midNifti
from dlh.data_management.utils import get_img_path_from_label_path, fetch_subject_and_session


## Variables
CONTRAST = {'t1': ['T1w'],
            't2': ['T2w'],
            't2s':['T2star'],
            't1_t2': ['T1w', 'T2w'],
            'psir': ['PSIR'],
            'stir': ['STIR'],
            'psir_stir': ['PSIR', 'STIR'],
            }

## Functions  
def extract_skeleton(inputs, outputs, norm_mean_skeleton, target=None, Flag_save=False, target_th=0.5):
    idtest = 1
    outputs  = outputs.data.cpu().numpy()
    if not target is None:
        target  = target.data.cpu().numpy()
    inputs = inputs.data.cpu().numpy()
    skeleton_images = []  # This variable stores an image to visualize discs 
    out_list = []
    for idx in range(outputs.shape[0]):    
        count_list = []
        Nch = outputs.shape[1]
        center_list = {}
        Final  = np.zeros((outputs.shape[0], Nch, outputs.shape[2], outputs.shape[3])) # Final array composed of the prediction (outputs) normalized and after applying a threshold       
        for idy in range(Nch):
            ych = outputs[idx, idy]
            #ych = np.rot90(ych)  # Rotate prediction to normal position
            ych = ych/np.max(np.max(ych))
            ych[np.where(ych<target_th)] = 0
            ych_fin = ych[:,:]
            ych = np.where(ych>0, 1.0, 0)
            ych = np.uint8(ych)
            num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(ych)
            count_list.append(num_labels-1) # Number of "object" in the image --> "-1" because we remove the backgroung
            center_list[str(idy)] = [t[::-1] for t in centers[1:]]
            Final[idx, idy] = ych_fin
            
        if np.prod(count_list)>25000:
            raise ValueError("Trop de possibilités")
        ups = []
        for c in count_list:
            ups.append(range(c))
        combs = cartesian(ups) # Create all the possible discs combinations
        best_loss = np.Inf
        best_skeleton = []
        for comb in combs:
            cnd_skeleton = []
            for joint_idx, cnd_joint_idx in enumerate(comb):
                cnd_center = center_list[str(joint_idx)][cnd_joint_idx]
                cnd_skeleton.append(cnd_center)
            loss = check_skeleton(cnd_skeleton, norm_mean_skeleton) # Compare each disc combination with norm_mean_skeleton
            if best_loss > loss:
                best_loss = loss
                best_skeleton = cnd_skeleton
        discs_idx = list(center_list.keys())
        Final2  = np.uint8(np.where(Final>0, 1, 0))  # Extract only non-zero values in the Final variable
        cordimg = np.zeros(Final2.shape)
        hits = np.zeros_like(outputs[0])
        for i, jp, in enumerate(best_skeleton): # Create an image with best skeleton
            jp = [int(t) for t in jp]
            hits[i, jp[0]-1:jp[0]+2, jp[1]-1:jp[1]+2] = [255, 255, 255]
            hits[i, :, :] = cv2.GaussianBlur(hits[i, :, :],(5,5),cv2.BORDER_DEFAULT)
            hits[i, :, :] = hits[i, :, :]/hits[i, :, :].max()*255
            cordimg[idx, i, jp[0], jp[1]] = 1
        
        for id_ in range(Final2.shape[1]):
            num_labels, labels_im = cv2.connectedComponents(Final2[idx, id_])
            for id_r in range(1, num_labels):
                if np.sum(np.sum((labels_im==id_r) * cordimg[idx, id_]) )>0:
                   labels_im = labels_im == id_r
                   continue
            Final2[idx, id_] = labels_im
        Final = Final * Final2
        
        out_dict = {}
        for i, disc_idx in enumerate(discs_idx):
            out_dict[str(int(disc_idx)+1)] = best_skeleton[i]          
        
        skeleton_images.append(hits)
        out_list.append(out_dict)
        
    skeleton_images = np.array(skeleton_images)
    if Flag_save:
        if not target is None: # TODO: Implement an else case to save images when no target are provided
            save_test_results(inputs, skeleton_images, targets=target, name=idtest, target_th=0.5)
            
    idtest+=1
    return Final, out_list

##
def check_skeleton(cnd_sk, mean_skeleton):
    cnd_sk = np.array(cnd_sk)
    Normjoint = np.linalg.norm(cnd_sk[0]-cnd_sk[4])
    for idx in range(1, len(cnd_sk)):
        cnd_sk[idx] = (cnd_sk[idx] - cnd_sk[0]) / Normjoint
    cnd_sk[0] -= cnd_sk[0]
    
    return np.sum(np.linalg.norm(mean_skeleton[:len(cnd_sk)]-cnd_sk))

##     
def save_test_results(inputs, outputs, targets, name='', target_th=0.5):
    clr_vis_Y = []
    hues = np.linspace(0, 179, targets.shape[1], dtype=np.uint8)
    blank_ch = 255*np.ones_like(targets[0,0], dtype=np.uint8)

    for Y in [targets, outputs]:
        for y, x in zip(Y, inputs):
            y_colored = np.zeros([y.shape[1], y.shape[2], 3], dtype=np.uint8)
            y_all = np.zeros([y.shape[1], y.shape[2]], dtype=np.uint8)
            
            for ych, hue_i in zip(y, hues):
                ych = ych/(np.max(np.max(ych))+0.0001)
                ych[np.where(ych<target_th)] = 0

                ych_hue = np.ones_like(ych, dtype=np.uint8)*hue_i
                ych = np.uint8(255*ych/(np.max(ych)+0.0001))

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

    
    t = np.array(clr_vis_Y)
    t = np.transpose(t, [0, 3, 1, 2])
    trgts = make_grid(torch.Tensor(t), nrow=4)

    txt = f'test/visualize/{name}_test_result.png'
    print(f'{txt} was created')
    res = np.transpose(trgts.numpy(), (1,2,0))
    cv2.imwrite(txt, res)

    
# looks for the closest points between real and predicted
def closest_node(node, nodes):
    nodes1 = np.asarray(nodes)
    dist_2 = np.sum((nodes1 - node) ** 2, axis=1)
    return np.argmin(dist_2), dist_2


##
def load_niftii_split(config_data, split='TRAINING'):
    '''
    This function output 5 lists corresponding to:
        - the middle slices extracted from the niftii images
        - the corresponding 2D masks with discs labels
        - the discs labels
        - the subjects names
        - the image slice shape
    
    :param config_data: Config dict where every label used for TRAINING, VALIDATION and/or TESTING has its path specified
    :param split: Split of the data needed ('TRAINING', 'VALIDATION', 'TESTING')
    '''

    # Check config type to ensure that labels paths are specified and not images
    if config_data['TYPE'] != 'LABEL':
        raise ValueError('TYPE LABEL not detected: PLZ specify paths to labels for training in config file')
    
    # Get file paths based on split
    label_paths = config_data[split]
    
    # Init progression bar
    bar = Bar(f'Load {split} data with pre-processing', max=len(label_paths))
    
    imgs = []
    masks = []
    discs_labels_list = []
    subjects = []
    shapes = []
    for label_path in label_paths:
        img_path = get_img_path_from_label_path(label_path)
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f'Error while loading subject\n {img_path} or {label_path} might not exist')
        else:
            # Applying preprocessing steps
            image, mask, discs_labels = apply_preprocessing(img_path, label_path)
            if discs_labels: # Check if file not empty
                imgs.append(image)
                masks.append(mask)
                discs_labels_list.append(discs_labels)
                subject, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(img_path)
                subjects.append(subject)
                shapes.append(get_midNifti(img_path).shape)
        
        # Plot progress
        bar.suffix  = f'{label_paths.index(label_path)+1}/{len(label_paths)}'
        bar.next()
    bar.finish()
    return imgs, masks, discs_labels_list, subjects, shapes

def load_img_only(config_data, split='TESTING'):
    '''
    This function output 3 lists corresponding to:
        - the middle slice extracted from the niftii images
        - the subjects names
        - the images shapes
    
    :param config_data: Config dict where every image used for TRAINING, VALIDATION and/or TESTING has its path specified
    :param split: Split of the data needed ('TRAINING', 'VALIDATION', 'TESTING')
    '''
    
    # Get file paths based on split
    paths = config_data[split]
    
    # Init progression bar
    bar = Bar(f'Load {split} data with pre-processing', max=len(paths))
    
    imgs = []
    subjects = []
    shapes = []
    for path in paths:
        # Check TYPE to get img_path
        if config_data['TYPE'] == 'IMAGE':
            img_path = path
        elif config_data['TYPE'] == 'LABEL':
            img_path = get_img_path_from_label_path(path)
        else:
            raise ValueError('TYPE error: The TYPE can only be "IMAGE" or "LABEL"')

        # Check if img_path exists
        if not os.path.exists(img_path):
            print(f'Error while loading subject\n {img_path} does not exist')
        else:
            # Applying preprocessing steps
            image = apply_preprocessing(img_path)
            imgs.append(image)
            subject, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(img_path)
            subjects.append(subject)
            shapes.append(get_midNifti(img_path).shape)
        
        # Plot progress
        bar.suffix  = f'{paths.index(path)+1}/{len(paths)}'
        bar.next()
    bar.finish()
    return imgs, subjects, shapes