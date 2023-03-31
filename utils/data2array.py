#===================================================
## Authors: 
# - Lucas Rouhier ()
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
#===================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib

from spinalcordtoolbox.image import Image

matplotlib.use("Agg")


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


def add_zero_padding(img_list, x_val=512, y_val=512):
    """
    Add zero padding to each image in an array so they all have matching dimension.
    :param img_list: list of input image to pad
    :param x_val: shape of output alongside x axis
    :param y_val: shape of output alongside y axis
    :return: list of images
    """
    if type(img_list) != list:
        img_list = [img_list]
    img_zero_padding_list = []
    for i in range(len(img_list)):
        img = img_list[i]
        img_tmp = np.zeros((x_val, y_val, 1), dtype=np.float64)
        img_tmp[0:img.shape[0], 0:img.shape[1], 0] = img
        img_zero_padding_list.append(img_tmp)

    return img_zero_padding_list


def mask2label(path_label, aim='full'):
    """
    Convert nifti image to a list of coordinates
    :param path_label: path of nifti image
    :return:
    """
    a = Image(path_label)
    return [list(coord) for coord in a.change_orientation('RPI').getNonZeroCoordinates(sorting='value')]


def get_midNifti(path_im, ind):
    """
    Retrieve the input images for the network. This images are generated by
    averaging the 7 slices in the middle of the volume
    :param path_im: path to image
    :param ind: index of the middle
    :return:
    """
    a = Image(path_im)
    a.change_orientation('RPI')
    arr = np.array(a.data)
    return np.mean(arr[ind - 3:ind + 3, :, :], 0)


def images_normalization(img_list, std=True):
    if type(img_list) != list:
        img_list = [img_list]
    img_norm_list = []
    for i in range(len(img_list)):
        # print('Normalizing ' + str(i + 1) + '/' + str(len(img_list)))
        img = img_list[i] - np.mean(img_list[i])  # zero-center
        if std:
            img_std = np.std(img)  # normalize
            epsilon = 1e-100
            img = img / (img_std + epsilon)  # epsilon is used in order to avoid by zero division
        img_norm_list.append(img)
    return img_norm_list


def load_Data_Bids2Array(DataSet_path, mode='t2', factor=0.9, split='train', aim='full'):
    """
    Load image into an array. array[0] will represent 2D images
    array[1] will be list of corresponding ground truth coordinates
    :param DataSet_path: Path to images
    :param mode: Mode t1 only load T1w , Mode t2 only load T2w , Different number load both
    :param split: Train or test. Decide which part of the dataset will be used.
    :return:
    """
    size_val = 512
    ds_image = []
    ds_label = []
    list_dir = os.listdir(DataSet_path)
    list_dir.sort()
    if '.DS_Store' in list_dir:
        list_dir.remove('.DS_Store')
    all_file = len(list_dir)
    if split == 'train':
        end = int(np.round(all_file * factor))
        begin = 0
    elif split == 'test':
        begin = int(np.round(all_file * factor))
        end = int(np.round(all_file * 1))
    for i in range(begin, end):
        path_tmp = os.path.join(DataSet_path,list_dir[i]) + '/'

        if mode != 't2':
            if os.path.exists(path_tmp + list_dir[i]+'_T1w_labels-disc-manual.nii.gz'):
                tmp_label = mask2label(path_tmp + list_dir[i]+'_T1w_labels-disc-manual.nii.gz',aim)
            else:
                continue
        if mode != 't1':
            print(path_tmp + list_dir[i])
            if os.path.exists(path_tmp + list_dir[i] +'_T2w_labels-disc-manual_r.nii.gz'):
                tmp_label_t2 = mask2label(path_tmp + list_dir[i]+'_T2w_labels-disc-manual_r.nii.gz',aim)
            elif os.path.exists(path_tmp + list_dir[i] +'_T2w_labels-disc-manual.nii.gz'):
                tmp_label_t2 = mask2label(path_tmp + list_dir[i]+'_T2w_labels-disc-manual.nii.gz',aim)
            else:
                continue

        if mode != 't1':
            index_mid = tmp_label_t2[0][0]
        else:
            index_mid = tmp_label[0][0]
        if mode != 't2':
            mid_slice = get_midNifti(path_tmp +list_dir[i]+ '_T1w.nii.gz', index_mid)
        if mode != 't1':
            if os.path.exists(path_tmp+list_dir[i]+'_acq-sag_T2w_r.nii.gz'):
                mid_slice_t2 = get_midNifti(path_tmp+list_dir[i]+'_acq-sag_T2w_r.nii.gz', index_mid)
            elif os.path.exists(path_tmp+list_dir[i]+'_T2w_r.nii.gz'):
                mid_slice_t2 = get_midNifti(path_tmp+list_dir[i]+'_T2w_r.nii.gz', index_mid)
            elif os.path.exists(path_tmp+list_dir[i]+'_T2w.nii.gz'):
                mid_slice_t2 = get_midNifti(path_tmp+list_dir[i]+'_T2w.nii.gz', index_mid)

        if mode == 't2':
            mid_slice = mid_slice_t2
        if split == 'train':
            if mid_slice.shape[0] > 450:
                print('removed')
                pass
            elif mid_slice.shape[1] > 450:
                print('removed')
                pass
            else:
                if mode != 't2':
                    ds_image.append(mid_slice)
                    ds_label.append(tmp_label)
                if mode != 't1':
                    ds_image.append(mid_slice_t2)
                    ds_label.append(tmp_label_t2)
        else:
            if mode != 't2':
                ds_image.append(mid_slice)
                ds_label.append(tmp_label)
            if mode != 't1':
                ds_image.append(mid_slice_t2)
                ds_label.append(tmp_label_t2)
    ds_image = images_normalization(ds_image)

    # Zero padding
    if 1:
        max_y = 0
        max_x = 0
        for i in range(len(ds_image)):
            # ds_image[i] = np.expand_dims(ds_image[i],-1)
            if ds_image[i].shape[1] > max_y:
                max_y = ds_image[i].shape[1]
            if ds_image[i].shape[0] > max_x:
                max_x = ds_image[i].shape[0]

    ds_image = add_zero_padding(ds_image, x_val=(int(np.ceil(max_x))), y_val=(int(np.ceil(max_y))))
    # val_ds_img = add_zero_padding(val_ds_img, x_val=size_val, y_val=size_val)
    # test_ds_img = add_zero_padding(test_ds_img, x_val=size_val, y_val=size_val)
    # Convert images to np.array
    # print(ds_image)
    # ds_image2 = np.array(ds_image)
    # print(ds_image.shape)

    return [ds_image, ds_label]

def load_Data_Bids2Array_with_subjects(DataSet_path, mode='t2', factor=0.9, split='train', aim='full'):
    """
    Load image into an array. array[0] will represent 2D images
    array[1] will be list of corresponding ground truth coordinates
    :param DataSet_path: Path to images
    :param mode: Mode t1 only load T1w , Mode t2 only load T2w , Different number load both
    :param split: Train or test. Decide which part of the dataset will be used.
    :return:
    """
    size_val = 512
    ds_image = []
    ds_label = []
    subjects_list = []
    list_dir = os.listdir(DataSet_path)
    list_dir.sort()
    if '.DS_Store' in list_dir:
        list_dir.remove('.DS_Store')
    all_file = len(list_dir)
    if split == 'train':
        end = int(np.round(all_file * factor))
        begin = 0
    elif split == 'test':
        begin = int(np.round(all_file * factor))
        end = int(np.round(all_file * 1))
    for i in range(begin, end):
        path_tmp = os.path.join(DataSet_path,list_dir[i]) + '/'
        subjects_list.append(list_dir[i])

        if mode != 't2':
            if os.path.exists(path_tmp + list_dir[i]+'_T1w_labels-disc-manual.nii.gz'):
                tmp_label = mask2label(path_tmp + list_dir[i]+'_T1w_labels-disc-manual.nii.gz',aim)
            else:
                continue
        if mode != 't1':
            print(path_tmp + list_dir[i])
            if os.path.exists(path_tmp + list_dir[i] +'_T2w_labels-disc-manual_r.nii.gz'):
                tmp_label_t2 = mask2label(path_tmp + list_dir[i]+'_T2w_labels-disc-manual_r.nii.gz',aim)
            elif os.path.exists(path_tmp + list_dir[i] +'_T2w_labels-disc-manual.nii.gz'):
                tmp_label_t2 = mask2label(path_tmp + list_dir[i]+'_T2w_labels-disc-manual.nii.gz',aim)
            else:
                continue

        if mode != 't1':
            index_mid = tmp_label_t2[0][0]
        else:
            index_mid = tmp_label[0][0]
        if mode != 't2':
            mid_slice = get_midNifti(path_tmp +list_dir[i]+ '_T1w.nii.gz', index_mid)
        if mode != 't1':
            if os.path.exists(path_tmp+list_dir[i]+'_acq-sag_T2w_r.nii.gz'):
                mid_slice_t2 = get_midNifti(path_tmp+list_dir[i]+'_acq-sag_T2w_r.nii.gz', index_mid)
            elif os.path.exists(path_tmp+list_dir[i]+'_T2w_r.nii.gz'):
                mid_slice_t2 = get_midNifti(path_tmp+list_dir[i]+'_T2w_r.nii.gz', index_mid)
            elif os.path.exists(path_tmp+list_dir[i]+'_T2w.nii.gz'):
                mid_slice_t2 = get_midNifti(path_tmp+list_dir[i]+'_T2w.nii.gz', index_mid)

        if mode == 't2':
            mid_slice = mid_slice_t2
        if split == 'train':
            if mid_slice.shape[0] > 450:
                print('removed')
                pass
            elif mid_slice.shape[1] > 450:
                print('removed')
                pass
            else:
                if mode != 't2':
                    ds_image.append(mid_slice)
                    ds_label.append(tmp_label)
                if mode != 't1':
                    ds_image.append(mid_slice_t2)
                    ds_label.append(tmp_label_t2)
        else:
            if mode != 't2':
                ds_image.append(mid_slice)
                ds_label.append(tmp_label)
            if mode != 't1':
                ds_image.append(mid_slice_t2)
                ds_label.append(tmp_label_t2)
    ds_image = images_normalization(ds_image)

    # Zero padding
    if 1:
        max_y = 0
        max_x = 0
        for i in range(len(ds_image)):
            # ds_image[i] = np.expand_dims(ds_image[i],-1)
            if ds_image[i].shape[1] > max_y:
                max_y = ds_image[i].shape[1]
            if ds_image[i].shape[0] > max_x:
                max_x = ds_image[i].shape[0]

    ds_image = add_zero_padding(ds_image, x_val=(int(np.ceil(max_x))), y_val=(int(np.ceil(max_y))))
    # val_ds_img = add_zero_padding(val_ds_img, x_val=size_val, y_val=size_val)
    # test_ds_img = add_zero_padding(test_ds_img, x_val=size_val, y_val=size_val)
    # Convert images to np.array
    # print(ds_image)
    # ds_image2 = np.array(ds_image)
    # print(ds_image.shape)

    return [ds_image, ds_label, subjects_list]