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
from sklearn.utils.extmath import cartesian
from torchvision.utils import make_grid
from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.utils.fs import tmp_create, rmtree
from spinalcordtoolbox.utils.sys import run_proc

## Variables
CONTRAST = {'t1': 'T1w',
            't2': 'T2w',
            't2s':'T2star'}

# Association between a vertebrae and the disc right above
VERT_DISC = {
    'C1':1,
    'C2':2,
    'C3':3,
    'C4':4,
    'C5':5,
    'C6':6,
    'C7':7,
    'T1':8,
    'T2':9,
    'T3':10,
    'T4':11,
    'T5':12,
    'T6':13,
    'T7':14,
    'T8':15,
    'T9':16,
    'T10':17,
    'T11':18,
    'T12':19,
    'L1':20,
    'L2':21,
    'L3':22,
    'L4':23,
    'L5':24,
    'S1':25
}

## Functions
def visualize_discs(input_img, coords_list, out_path):
    coords_list = swap_y_origin(coords=coords_list, img_shape=input_img.shape, y_pos=0).tolist() # The y origin is at the top of the image
    discs_images = []
    for coord in coords_list:
        coord = [int(c) for c in coord]
        disc_img = np.zeros_like(input_img[:,:])
        disc_img[coord[0]-1:coord[0]+2, coord[1]-1:coord[1]+2] = [255, 255, 255]
        disc_img[:, :] = cv2.GaussianBlur(disc_img[:, :],(5,5),cv2.BORDER_DEFAULT)
        disc_img[:, :] = disc_img[:, :]/disc_img[:, :].max()*255
        discs_images.append(disc_img)
    discs_images = np.array(discs_images)
    save_discs_image(input_img, discs_images, out_path)

##    
def extract_skeleton(inputs, outputs, target, norm_mean_skeleton, ndiscs, Flag_save = False, target_th=0.5):
    idtest = 1
    outputs  = outputs.data.cpu().numpy()
    target  = target.data.cpu().numpy()
    inputs = inputs.data.cpu().numpy()
    skeleton_images = []  # This variable stores an image to visualize discs 
    for idx in range(outputs.shape[0]):    
        count_list = []
        Nch = 0
        center_list = {}
        while np.sum(np.sum(target[idx, Nch]))>0 and Nch<target.shape[1]:
            Nch += 1
        if Nch>=target.shape[1]:
            print(f'The method is able to detect {ndiscs} discs, more discs may be present in the image')
        Final  = np.zeros((outputs.shape[0], Nch, outputs.shape[2], outputs.shape[3])) # Final array composed of the prediction (outputs) normalized and after applying a threshold       
        for idy in range(Nch): 
            ych = outputs[idx, idy]
            ych = np.rot90(ych)  # Rotate prediction to normal position
            ych = ych/np.max(np.max(ych))
            ych[np.where(ych<target_th)] = 0
            Final[idx, idy] = ych
            ych = np.where(ych>0, 1.0, 0)
            ych = np.uint8(ych)
            num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(ych)
            count_list.append(num_labels-1)
            center_list[str(idy)] = [t[::-1] for t in centers[1:]]
            
        ups = []
        for c in count_list:
            ups.append(range(c))
        combs = cartesian(ups)
        best_loss = np.Inf
        best_skeleton = []
        for comb in combs:
            cnd_skeleton = []
            for joint_idx, cnd_joint_idx in enumerate(comb):
                cnd_center = center_list[str(joint_idx)][cnd_joint_idx]
                cnd_skeleton.append(cnd_center)
            loss = check_skeleton(cnd_skeleton, norm_mean_skeleton)
            if best_loss > loss:
                best_loss = loss
                best_skeleton = cnd_skeleton
        Final2  = np.uint8(np.where(Final>0, 1, 0))  # Extract only non-zero values in the Final variable
        cordimg = np.zeros(Final2.shape)
        hits = np.zeros_like(outputs[0])
        for i, jp, in enumerate(best_skeleton):
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
                
        
        skeleton_images.append(hits)
        
    skeleton_images = np.array(skeleton_images)
    inputs = np.rot90(inputs, axes=(-2, -1))  # Rotate input to normal position
    target = np.rot90(target, axes=(-2, -1))  # Rotate target to normal position
    if Flag_save:
      save_test_results(inputs, skeleton_images, targets=target, name=idtest, target_th=0.5)
    idtest+=1
    return Final

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
                ych = ych/np.max(np.max(ych))
                ych[np.where(ych<target_th)] = 0
                # ych = cv2.GaussianBlur(ych,(15,15),cv2.BORDER_DEFAULT)

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
            # img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
            clr_vis_Y.append(img_mix)

    
    t = np.array(clr_vis_Y)
    t = np.transpose(t, [0, 3, 1, 2])
    trgts = make_grid(torch.Tensor(t), nrow=4)

    txt = f'test/visualize/{name}_test_result.png'
    res = np.transpose(trgts.numpy(), (1,2,0))
    cv2.imwrite(txt, res)

##
def save_discs_image(input_img, discs_images, out_path, target_th=0.5):
    clr_vis_Y = []
    hues = np.linspace(0, 179, discs_images.shape[0], dtype=np.uint8)
    blank_ch = 255*np.ones_like(discs_images[0], dtype=np.uint8)

    y_colored = np.zeros([discs_images.shape[1], discs_images.shape[2], 3], dtype=np.uint8)
    y_all = np.zeros([discs_images.shape[1], discs_images.shape[2]], dtype=np.uint8)
    
    for ych, hue_i in zip(discs_images, hues):
        ych = ych/np.max(np.max(ych))

        ych_hue = np.ones_like(ych, dtype=np.uint8)*hue_i
        ych = np.uint8(255*ych/np.max(ych))

        colored_ych = np.zeros_like(y_colored, dtype=np.uint8)
        colored_ych[:, :, 0] = ych_hue
        colored_ych[:, :, 1] = blank_ch
        colored_ych[:, :, 2] = ych
        colored_y = cv2.cvtColor(colored_ych, cv2.COLOR_HSV2BGR)

        y_colored += colored_y
        y_all += ych

    # Normalised image between [0,255] as integer
    x = (255*(input_img - np.min(input_img))/np.ptp(input_img)).astype(int)

    x_3ch = np.zeros([x.shape[0], x.shape[1], 3])
    for i in range(3):
        x_3ch[:, :, i] = x[:, :]

    img_mix = np.uint8(x_3ch*0.6 + y_colored*0.4)
    clr_vis_Y.append(img_mix)
            
    t = np.array(clr_vis_Y)
    t = np.transpose(t, [0, 3, 1, 2])
    trgts = make_grid(torch.Tensor(t), nrow=4)

    res = np.transpose(trgts.numpy(), (1,2,0))
    cv2.imwrite(out_path, res)
    
# looks for the closest points between real and predicted
def closest_node(node, nodes):
    nodes1 = np.asarray(nodes)
    dist_2 = np.sum((nodes1 - node) ** 2, axis=1)
    return np.argmin(dist_2), dist_2

# looks for the best association between ground truth and predicted discs
def best_disc_association(pred, gt):
    '''
    pred: numpy array of the coordinate of M discs
    gt: numpy array of the coordinate of N discs + num of the discs
    Note: M and N can be different
    
    return: Two lists (pred, gt) with the same length L corresponding to the biggest disc number in
    ground truth: L = gt[:,-1].max()
    '''
    M = pred.shape[0]
    N = gt.shape[0]
    L = gt[:,-1].max()
    pred_out, gt_out = [0]*L, [0]*L
    #if N >= M:
    dist_m = []
    for m in range(M):
        dist_m.append(np.sum((gt[:,:2] - pred[m]) ** 2, axis=1))
    dist_m = np.array(dist_m)
    ref_coord = []
    for n in range(N):
        disc_num = gt[n,-1]
        closer_to_node_n = np.argmin(dist_m[:,n])
        ref_coord.append([disc_num, closer_to_node_n, dist_m[closer_to_node_n,n]])
    ref_coord = np.array(ref_coord)
    new_ref_coord = []
    pred_coord_list = []
    for i in ref_coord:
        node_repetition = np.where((ref_coord[:,1]==i[1]))
        node = node_repetition[0][0]
        min_dist_node = ref_coord[node,2]
        if len(node_repetition[0]) > 1:
            for j in node_repetition[0][1:]:
                if ref_coord[j,2]<min_dist_node:
                    min_dist_node = ref_coord[j,2]
                    node = j
        if ref_coord[node,1] not in pred_coord_list:
            new_ref_coord.append(ref_coord[node])
            pred_coord_list.append(ref_coord[node,1])
    if len(pred_coord_list)<M:  # Every prediction point is not referenced
        for k in range(M):
            if k not in pred_coord_list:
                node, dist = closest_node(pred[k],gt[:,:2])
                closest_disc_num = gt[node,-1]
                if (closest_disc_num + 1) not in new_ref_coord[:][0]:
                    disc_num = closest_disc_num + 1
                    new_ref_coord.append([disc_num, k, dist])
                    
                elif (closest_disc_num - 1) not in new_ref_coord[:][0]:
                    disc_num = closest_disc_num - 1
                    new_ref_coord.append([disc_num, k, dist])
                    
                else:
                    print('Prediction disc error: discs might be misplaced, check disc:',closest_disc_num)
                
        
    if M > N: # TODO check this condition
        print('More discs detected by hourglass')
        print('nb_gt', N)
        print('nb_hourglass', M)
        print('PLZ CHECK THE SCRIPT')
        for j in range(M):
            if j not in new_ref_coord[:,1]: # Let's assume it's an extremity disc
                closest_gt, dist = closest_node(pred[j],gt[:,:2])
                if pred[j][0] < gt[closest_gt][0]:
                    disc_num = gt[closest_gt][-1] + 1
                    np.append(ref_coord,np.array([disc_num, j, dist]))
                else:
                    disc_num = gt[closest_gt][-1] - 1
                    if disc_num >= 1:
                        np.append(ref_coord,np.array([disc_num, j, dist]))
                    else:
                        print('Impossible disc prediction')
        for n in range(N):
            disc_num = int(gt[n,-1])
            gt_out[disc_num-1]=gt[n].tolist()
    else:
        
        for i in range(len(gt)):
            disc_num = gt[i][-1]
            gt_out[disc_num-1] = gt[i].tolist()
        
    for disc_num, closer_to_node_n, dist_m in new_ref_coord:
        disc_num = int(disc_num)
        closer_to_node_n = int(closer_to_node_n)      
        pred_out[disc_num-1]=pred[closer_to_node_n].tolist()
            
    return pred_out, gt_out

##
def swap_y_origin(coords, img_shape, y_pos=1):
    '''
    This function returns a list of coords where the y origin coords was moved from top and bottom
    '''
    y_shape = img_shape[1]
    coords[:,y_pos] = y_shape - coords[:,y_pos]
    return coords

##
def coord2list(coords):
    '''
    This function swaps between coordinate referencial and list referencial
    [x, y] <--> [lines columns]
    '''
    return np.flip(coords,axis=1)

##
def str2array(coords):
    '''
    coords: numpy array of str coords
    
    returns numpy array of specified coordinates and None for not specified coordinates
    '''
    output_coords = []
    for coord in coords:
        if coord not in ['Fail', 'None', 'None\n']:
            coord_split = coord.split(',')
            output_coords.append([float(coord_split[0].split('[')[1]),float(coord_split[1].split(']')[0])])
        else:
            output_coords.append(None)
    return np.array(output_coords)

## 
def check_missing_discs(coords):
    '''
    coords: numpy array of coords
    return:
        output_coords: list of coords without missing discs
        missing_discs: list of missing discs detection
    '''
    output_coords = []
    missing_discs = []
    for num, coord in enumerate(coords):
        if coord is not None:
            output_coords.append(coord)
        else:
            missing_discs.append(num+1)
    return np.array(output_coords), np.array(missing_discs)
            
##
def project_on_spinal_cord(coords, seg_path, disc_num=True, proj_2d=False):
    '''
    This function projects discs coordinates onto the centerline using sct_label_utils
    
    coords: numpy array of coords
    return: numpy array of coords projected on the spinal cord
    '''
    # Get segmentation
    fname_seg = os.path.abspath(seg_path)
    seg = Image(fname_seg).change_orientation('RPI')
    
    # Create temp folder
    path_tmp = tmp_create(basename="label-vertebrae-project")
    
    if proj_2d:
        # Get middle slice of the image because last coordinate is missing to create a 3d object
        middle_slice = np.array([seg.data.shape[0]//2]*coords.shape[0])

    # Get transpose
    coords_t = np.rint(np.transpose(coords)).astype(int)
    
    # Create Image object of the referenced coordinates
    fname_coords = zeros_like(seg)
    
    if proj_2d:
        if disc_num:
            fname_coords.data[middle_slice, coords_t[0], coords_t[1]] = coords_t[2]
        else:
            fname_coords.data[middle_slice, coords_t[0], coords_t[1]] = 1
    else:
        if disc_num:
            fname_coords.data[coords_t[0], coords_t[1], coords_t[2]] = coords_t[3]
        else:
            fname_coords.data[coords_t[0], coords_t[1], coords_t[2]] = 1
    
    # Save referenced coords into temp folder
    discs_path = os.path.join(path_tmp, 'labels.nii.gz')
    fname_coords.save(path=discs_path, dtype='float32')
    
    # Compute projection
    out_path = os.path.join(path_tmp, 'proj_labels.nii.gz')
    status, _ = run_proc(['sct_label_utils',
                            '-i', fname_seg,
                            '-project-centerline', discs_path,
                            '-o', out_path], raise_exception=False)
    if status == 0:
        if disc_num:
            discs_coords = Image(out_path).getNonZeroCoordinates(sorting='value')
        else:
            discs_coords = Image(out_path).getNonZeroCoordinates(sorting='z', reverse_coord=True)
    else:
        print('Fail projection')
        
    if proj_2d:    
        if disc_num:
            output_coords = np.array([list(coord)[1:] for coord in discs_coords])
        else:
            output_coords = np.array([list(coord)[1:3] for coord in discs_coords])
    else:
        if disc_num:
            output_coords = np.array([list(coord) for coord in discs_coords])
        else:
            output_coords = np.array([list(coord)[:3] for coord in discs_coords])
    
    # Remove temporary folder
    rmtree(path_tmp)

    return output_coords