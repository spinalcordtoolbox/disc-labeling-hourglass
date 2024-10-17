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
import matplotlib.pyplot as plt
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
            't1_t2_psir_stir': ['T1w', 'T2w', 'PSIR', 'STIR']
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
            if (num_labels-1) * np.prod(count_list) < 1000 and (num_labels-1) < 4: # Remove masks with too many predictions
                count_list.append(num_labels-1) # Number of "object" in the image --> "-1" because we remove the backgroung
                center_list[str(idy)] = [t[::-1] for t in centers[1:]]
                Final[idx, idy] = ych_fin
        
        if not count_list:
            raise ValueError("Trop de possibilités")
        
        ups = []
        for c in count_list:
            ups.append(range(c))
        combs = cartesian(ups) # Create all the possible discs combinations
        best_loss = np.Inf
        best_skeleton = []
        for comb in combs:
            cnd_skeleton = []
            for joint_idx, cnd_joint_idx in zip(center_list.keys(), comb):
                cnd_center = center_list[joint_idx][cnd_joint_idx]
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
        z_coord = 0
        for i, disc_idx in enumerate(discs_idx):
            if best_skeleton[i][0] > z_coord: # Track z coordinate to ensure increasing disc value
                out_dict[str(int(disc_idx)+1)] = best_skeleton[i]          
                z_coord = best_skeleton[i][0]
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
def load_niftii_split(config_data, num_channel, fov=None, split='TRAINING'):
    '''
    This function output 5 lists corresponding to:
        - the middle slices extracted from the niftii images
        - the corresponding 2D masks with discs labels
        - the discs labels
        - the subjects names
        - image resolutions
        - the image slice shape
    
    :param config_data: Config dict where every label used for TRAINING, VALIDATION and/or TESTING has its path specified
    :param split: Split of the data needed ('TRAINING', 'VALIDATION', 'TESTING')
    '''

    # Check config type to ensure that labels paths are specified and not images
    if config_data['TYPE'] != 'LABEL':
        raise ValueError('TYPE LABEL not detected: PLZ specify paths to labels for training in config file')
    
    # Get file paths based on split
    paths = config_data[split]
    
    # Init progression bar
    bar = Bar(f'Load {split} data with pre-processing', max=len(paths))
    
    imgs = []
    masks = []
    discs_labels_list = []
    subjects = []
    shapes = []
    resolutions = []
    problematic_gt = []
    for dic in paths:
        img_path = os.path.join(config_data['DATASETS_PATH'], dic['IMAGE'])
        label_path = os.path.join(config_data['DATASETS_PATH'], dic['LABEL'])
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f'Error while loading subject\n {img_path} or {label_path} might not exist')
        else:
            # Applying preprocessing steps
            image, mask, discs_labels, res_image, shape_image = apply_preprocessing(img_path, label_path, num_channel)
            # Calculate number of images to add based on cropped fov
            if not fov is None:
                Y, X = round(fov[1]/res_image[0]), round(fov[0]/res_image[1])
                nb_same_img = shape_image[0]//Y + shape_image[0]//X + 3
            else:
                nb_same_img = 1
            if discs_labels and (max(np.array(discs_labels)[:,-1])+1-min(np.array(discs_labels)[:,-1]) == len(np.array(discs_labels))) and (np.array(discs_labels)[:,1] == np.sort(np.array(discs_labels)[:,1])).all(): # Check if file not empty or missing discs
                for i in range(nb_same_img): # Add the same image and masks nb_same_img times when random fov/crop is used
                    imgs.append(image)
                    masks.append(mask)
                    discs_labels_list.append(discs_labels)
                    subject, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(img_path)
                    subjects.append(subject)
                    resolutions.append(res_image)
                    shapes.append(shape_image)
            else:
                problematic_gt.append(label_path)
        
        # Plot progress
        bar.suffix  = f'{paths.index(dic)+1}/{len(paths)}'
        bar.next()
    bar.finish()

    # plot discs distribution
    plot_discs_distribution(discs_labels_list, out_path=f'discs_distribution_{split}.png')

    if problematic_gt:
        print("Error with these ground truth\n" + '\n'.join(problematic_gt))
    return imgs, masks, discs_labels_list, subjects, resolutions, shapes

def load_img_only(config_data, split='TESTING'):
    '''
    This function output 3 lists corresponding to:
        - the middle slice extracted from the niftii images
        - the subjects names
        - the images shapes
        - image resolutions
    
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
    resolutions = []
    for dic in paths:
        img_path = os.path.join(config_data['DATASETS_PATH'], dic['IMAGE'])

        # Check if img_path exists
        if not os.path.exists(img_path):
            print(f'Error while loading subject\n {img_path} does not exist')
        else:
            # Applying preprocessing steps
            image, res_image, shape_image = apply_preprocessing(img_path)
            imgs.append(image)
            subject, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(img_path)
            subjects.append(subject)
            resolutions.append(res_image)
            shapes.append(shape_image)
        
        # Plot progress
        bar.suffix  = f'{paths.index(dic)+1}/{len(paths)}'
        bar.next()
    bar.finish()
    return imgs, subjects, shapes, resolutions


def save_bar(names, values, output_path, x_axis, y_axis):
    '''
    Create a histogram plot
    :param names: String list of the names
    :param values: Values associated with the names
    :param output_path: Output path (string)
    :param x_axis: x-axis name
    :param y_axis: y-axis name

    '''
            
    # Set position of bar on X axis
    fig = plt.figure(figsize = (len(names)//2, 5))
 
    # creating the bar plot
    plt.bar(names, values, width = 0.4)
    
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xticks(names)
    plt.title("Discs distribution")
    plt.savefig(output_path)


def plot_discs_distribution(discs_labels_list, out_path):
    plot_discs = {}
    for discs_list in discs_labels_list:
        for disc_coords in discs_list:
            num_disc = disc_coords[-1]
            if not num_disc in plot_discs.keys():
                plot_discs[num_disc] = 1
            else:
                plot_discs[num_disc] += 1
    # Sort dict
    plot_discs = dict(sorted(plot_discs.items()))
    names, values = list(plot_discs.keys()), list(plot_discs.values())
    # Plot distribution
    save_bar(names, values, out_path, x_axis='Discs number', y_axis='Quantity')
    