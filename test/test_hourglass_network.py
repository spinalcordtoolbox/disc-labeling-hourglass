import os
import sys
import cv2
import torch
import pickle
import argparse
import numpy as np
from torch.utils.data import DataLoader

from dlh.models.hourglass import hg
from dlh.models.atthourglass import atthg
from dlh.utils.train_utils import image_Dataset
from dlh.utils.test_utils import CONTRAST, extract_skeleton, swap_y_origin, project_on_spinal_cord, edit_subject_lines_txt_file, load_niftii_split

#---------------------------Test Hourglass Network----------------------------
def test_hourglass(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrast = CONTRAST[args.contrast]
    train_contrasts = args.train_contrasts
    ndiscs = args.ndiscs
    origin_data = args.datapath
    skeleton_dir = args.skeleton_dir
    txt_file = args.out_txt_file
    label_suffix = args.suffix_label
    img_suffix = args.suffix_img
    
    # Error if multiple contrasts
    if len(contrast) > 1:
        print(f'Only one contrast may be selected with args.contrast, {len(contrast)} were selected')
        sys.exit(1)
    
    # Loading image paths
    print('loading images...')
    imgs_test, masks_test, discs_labels_test, subjects_test, original_shapes = load_niftii_split(datapath=origin_data, 
                                                                                                contrasts=contrast, 
                                                                                                split='test', 
                                                                                                split_ratio=(0.8, 0.1, 0.1),
                                                                                                label_suffix=label_suffix,
                                                                                                img_suffix=img_suffix)
    norm_mean_skeleton = np.load(os.path.join(skeleton_dir, f'{train_contrasts}_Skelet_ndiscs_{ndiscs}.npy'))
    
    # Load network weights
    if args.att:
        model = atthg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=ndiscs)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'src/dlh/weights/model_{train_contrasts}_att_stacks_{args.stacks}_ndiscs_{ndiscs}', map_location='cpu')['model_weights'])
    else:
        model = hg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=ndiscs)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(f'src/dlh/weights/model_{train_contrasts}_stacks_{args.stacks}_ndiscs_{ndiscs}', map_location='cpu')['model_weights'])

    # Create Dataloader
    full_dataset_test = image_Dataset(images=imgs_test, 
                                       targets=masks_test,
                                       discs_labels_list=discs_labels_test,
                                       subjects_names=subjects_test,
                                       num_channel=args.ndiscs,
                                       use_flip = False,
                                       load_mode='test'
                                      ) 
    MRI_test_loader   = DataLoader(full_dataset_test, 
                                   batch_size= 1, 
                                   shuffle=False, 
                                   num_workers=0
                                   )
    model.eval()
    
    # Load disc_coords txt file
    with open(txt_file,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    # Extract discs coordinates from the test set
    print('Processing with hourglass')
    for i, (inputs, targets, vis, gt_coord, subject_name) in enumerate(MRI_test_loader): # subject_name
        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
        output = model(inputs) 
        output = output[-1]
        subject_name = subject_name[0]
        
        prediction, pred_discs_coords = extract_skeleton(inputs, output, targets, norm_mean_skeleton, ndiscs, Flag_save=True)
        
        # Convert pred_discs_coords to original image size
        pred_shape = prediction[0,0].shape 
        original_shape = original_shapes[i] 
        pred = np.array([[(round(coord[0])/pred_shape[0])*original_shape[0], (round(coord[1])/pred_shape[1])*original_shape[1], int(disc_num)] for disc_num, coord in pred_discs_coords[0].items()]).astype(int)
        
        # Project coordinate onto the spinal cord centerline
        seg_path = os.path.join(origin_data, subject_name, f'{subject_name}_{contrast[0]}_seg.nii.gz' )
        pred = project_on_spinal_cord(coords=pred, seg_path=seg_path, disc_num=True, proj_2d=True)
        
        # Swap axis prediction and ground truth
        pred = swap_y_origin(coords=pred, img_shape=original_shape, y_pos=0).astype(int)  # Move y origin to the bottom of the image like Niftii convention
        
        # Edit coordinates in txt file
        # line = subject_name contrast disc_num gt_coords sct_discs_coords hourglass_coords spinenet_coords
        split_lines = edit_subject_lines_txt_file(coords=pred, txt_lines=split_lines, subject_name=subject_name, contrast=contrast[0], method_name='hourglass_coords')
                
    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run test on hourglass')

    parser.add_argument('--datapath', default="/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/preprocessed_data/vertebral_data", type=str,
                        help='dataset path')                                                             
    parser.add_argument('-c', '--contrast', default='t2', type=str, metavar='N',
                        help='MRI contrast of the tested images defaut=t2, only one contrast is possible here')
    parser.add_argument('--train-contrasts', default=parser.parse_args().contrast, type=str, metavar='N',
                        help='MRI contrast used for the training default=contrast parameter, multiple contrasts are allowed')
    parser.add_argument('--ndiscs', type=int, default=15,
                        help='Number of discs to detect')
    parser.add_argument('-txt', '--out-txt-file', default=os.path.join('test/files', f'{CONTRAST[parser.parse_args().contrast]}_hg{parser.parse_args().ndiscs}_discs_coords.txt'),
                        type=str, metavar='N',help='Generated txt file')
    parser.add_argument('--skeleton-dir', default=os.path.join(parser.parse_args().datapath, 'skeletons'),
                        type=str, metavar='N',help='Generated txt file')
    parser.add_argument('-sub', default= 'sub-perform04',
                        type=str, metavar='N',help='Generated txt file') # 'sub-juntendo750w06'
    
    parser.add_argument('--att', default= True, type=bool,
                        help=' Use attention mechanism') 
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    
    test_hourglass(parser.parse_args())