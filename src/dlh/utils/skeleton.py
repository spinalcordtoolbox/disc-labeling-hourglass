from __future__ import print_function, absolute_import
import os
import sys
import argparse
import numpy as np
import json
from torch.utils.data import DataLoader 
import cv2

from dlh.utils.image_dataset import image_Dataset
from dlh.utils.test_utils import CONTRAST, load_niftii_split

def create_skeleton(args):
    '''
    Create Skelet file to improve disc estimation of the hourglass network
    '''
 
    ndiscs = args.ndiscs 
    out_dir = args.skeleton_folder
    
    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)
    
    # Fetch contrast info from config data
    contrast_str = config_data['CONTRASTS'] # contrast_str is a unique string representing all the contrasts

    # Create skeleton folder to store training skeletons
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Loading images for training
    print('loading images...')
    imgs_train, masks_train, discs_labels_train, subjects_train, res_train, _ = load_niftii_split(config_data=config_data,
                                                                                       num_channel=args.ndiscs,
                                                                                       split='TRAINING')

    ## Create a dataset loader
    full_dataset_train = image_Dataset(images=imgs_train, 
                                       targets=masks_train,
                                       discs_labels=discs_labels_train,
                                       img_res=res_train,
                                       subjects_names=subjects_train,
                                       num_channel=args.ndiscs,
                                       use_flip = False,
                                       load_mode='val'
                                       )
    
    MRI_train_loader = DataLoader(full_dataset_train,
                                  batch_size= 1, 
                                  shuffle=False, 
                                  num_workers=0
                                  )

    All_skeletons = np.zeros((len(MRI_train_loader), ndiscs, 2))
    Joint_counter = np.zeros((ndiscs, 1))
    for i, (inputs, target, vis) in enumerate(MRI_train_loader):
        target = target.numpy()
        mask = np.zeros((target.shape[2], target.shape[3]))
        for idc in range(target.shape[1]):
            mask += target[0, idc]
        mask = np.uint8(np.where(mask>0, 1, 0))
        num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(mask)
        centers = [t[::-1] for t in centers]
        skelet = np.zeros((ndiscs, 2))
        skelet[0:len(centers)-1] = centers[1:]
        Normjoint = np.linalg.norm(skelet[0]-skelet[4]) # TODO: fix normalization issue https://github.com/spinalcordtoolbox/disc-labeling-hourglass/issues/29
        for idx in range(1, len(centers)-1):
            skelet[idx] = (skelet[idx] - skelet[0]) / Normjoint

        skelet[0] *= 0
        
        All_skeletons[i] = skelet
        Joint_counter[0:len(centers)-1] += 1
        
    Skelet = np.sum(All_skeletons, axis= 0)   
    Joint_counter[Joint_counter==0]=1  # To avoid dividing by zero
    Skelet /= Joint_counter

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    np.save(os.path.join(out_dir, f'{contrast_str}_Skelet_ndiscs_{ndiscs}.npy'), Skelet)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create hourglass skeleton')
    
    ## Parameters
    parser.add_argument('--config-data', type=str,
                        help='Config JSON file where every label used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    
    parser.add_argument('--ndiscs', type=int, default=25,
                        help='Number of discs to detect (default=25)')
    parser.add_argument('--skeleton-folder', type=str, default=os.path.abspath('src/dlh/skeletons'),
                        help='Folder where skeletons are stored. Will be created if does not exist. (default="src/dlh/skeletons")')
    
    create_skeleton(parser.parse_args())  # Create skeleton file to improve hourglass accuracy during testing
