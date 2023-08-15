from __future__ import print_function, absolute_import
import os
import sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import json
from torch.utils.data import DataLoader 
import cv2

from dlh.utils.train_utils import image_Dataset
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
    imgs_train, masks_train, discs_labels_train, subjects_train, _ = load_niftii_split(config_data=config_data, 
                                                                                   split='TRAIN')

    ## Create a dataset loader
    full_dataset_train = image_Dataset(images=imgs_train, 
                                       targets=masks_train,
                                       subjects_names=subjects_train,
                                       num_channel=args.ndiscs,
                                       use_flip = True,
                                       load_mode='val'
                                       )
    
    MRI_train_loader = DataLoader(full_dataset_train,
                                  batch_size= 1, 
                                  shuffle=False, 
                                  num_workers=0
                                  )

    All_skeletons = np.zeros((len(MRI_train_loader), ndiscs, 2))
    Joint_counter = np.zeros((ndiscs, 1))
    for i, (input, target, vis) in enumerate(MRI_train_loader):
        target = target.numpy()
        mask = np.zeros((target.shape[2], target.shape[3]))
        for idc in range(target.shape[1]):
            mask += target[0, idc]
        mask = np.uint8(np.where(mask>0, 1, 0))
        #mask = np.rot90(mask)
        num_labels, labels_im, states, centers = cv2.connectedComponentsWithStats(mask)
        centers = [t[::-1] for t in centers]
        skelet = np.zeros((ndiscs, 2))
        skelet[0:len(centers)-1] = centers[1:]
        Normjoint = np.linalg.norm(skelet[0]-skelet[4])
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
    parser = argparse.ArgumentParser(description='Training hourglass network')
    
    ## Parameters
    parser.add_argument('--datapath', type=str, required=True,
                        help='Path to data folder generated using data_management/gather_data.py Example: ~/<your_dataset>/vertebral_data (Required)')
    parser.add_argument('-c', '--contrasts', type=str, metavar='N', required=True,
                        help='MRI contrast: choices=["t1", "t2", "t1_t2"] (Required)')
    
    parser.add_argument('--ndiscs', type=int, default=11,
                        help='Number of discs to detect (default=11)')
    parser.add_argument('--skeleton-folder', type=str, default=os.path.abspath('src/dlh/skeletons'),
                        help='Folder where skeletons are stored. Will be created if does not exist. (default="src/dlh/skeletons")')
    parser.add_argument('--split-ratio', default=(0.8, 0.1, 0.1),
                        help='Split ratio used for (train, val, test) (default=(0.8, 0.1, 0.1))')
    
    create_skeleton(parser.parse_args())  # Create skeleton file to improve hourglass accuracy during testing
