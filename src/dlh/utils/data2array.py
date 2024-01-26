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

import numpy as np
import matplotlib

from dlh.utils.image import Image

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


def mask2label(path_label):
    """
    Convert nifti image to a list of coordinates
    :param path_label: path of nifti image
    :return:
    """
    a = Image(path_label).change_orientation('RSP')
    nx, ny, nz, nt, px, py, pz, pt = a.dim
    return [list(coord) for coord in a.getNonZeroCoordinates(sorting='value')], (py, pz), a.data.shape


def get_midNifti(path_im):
    """
    Fetch and average the middle slices of a 3D image.
    :param path_im: path to image
    :return: averaged image and sagittal resolutions
    """
    a = Image(path_im).change_orientation('RSP')
    nx, ny, nz, nt, px, py, pz, pt = a.dim
    nb_slice = 1
    while (nb_slice + 2)*px < 3: # To avoid having too blurry images
        nb_slice += 2 # Add 2 to always be symetrical

    arr = np.array(a.data)
    ind = arr.shape[0]//2
    if nb_slice > 1:
        out_img = np.mean(arr[ind - (nb_slice-1)//2:ind + (nb_slice-1)//2, :, :], 0)
    else:
        out_img = arr[ind, :, :]
    
    return out_img, (py, pz), a.data.shape


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
