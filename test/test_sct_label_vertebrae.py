import os
import numpy as np
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import run_proc

from dlh.utils.test_utils import CONTRAST, edit_subject_lines_txt_file


#---------------------------Test Sct Label Vertebrae--------------------------
def test_sct_label_vertebrae(args):
    '''
    Use sct_deepseg_sc and sct_label_vertebrae to find the vertebrae discs coordinates and append them
    to a txt file
    '''
    datapath = os.path.abspath(args.datapath)
    contrast = CONTRAST[args.contrast][0]
    txt_file = args.out_txt_file
    img_suffix = args.suffix_img
    
    # Extract txt file lines
    with open(txt_file,"r") as f:
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    print('Processing with sct_label_vertebrae')
    for dir_name in os.listdir(datapath):
        if dir_name.startswith('sub'):
            file_name = dir_name + img_suffix + '_' + contrast + '.nii.gz'
            file_path = os.path.join(datapath, dir_name, file_name)  # path to the original image
            seg_path = file_path.replace('.nii.gz', '_seg.nii.gz')  # path to the spinal cord segmentation
            if os.path.exists(seg_path):
                pass
            else:
                status, _ = run_proc(['sct_deepseg_sc',
                                        '-i', file_path, 
                                        '-c', args.contrast,
                                        '-o', seg_path])
                if status != 0:
                    print('Fail segmentation')
                    discs_coords = np.array([]) # Fail
            
            disc_file_path = file_path.replace('.nii.gz', '_seg_labeled_discs.nii.gz')  # path to the file with disc labels
            if os.path.exists(disc_file_path):
                # retrieve all disc coords
                discs_coords = np.array([list(coord) for coord in Image(disc_file_path).change_orientation("RIP").getNonZeroCoordinates(sorting='value')]).astype(int)
                # keep only 2D coordinates
                discs_coords = discs_coords[:, 1:]
            else:
                status, _ = run_proc(['sct_label_vertebrae',
                                            '-i', file_path,
                                            '-s', file_path.replace('.nii.gz', '_seg.nii.gz'),
                                            '-c', args.contrast,
                                            '-ofolder', os.path.join(datapath, dir_name)], raise_exception=False)
                if status == 0:
                    discs_coords = np.array([list(coord) for coord in Image(disc_file_path).change_orientation("RIP").getNonZeroCoordinates(sorting='value')]).astype(int)
                    # keep only 2D coordinates
                    discs_coords = discs_coords[:, 1:]         
                else:
                    print('Exit value 1')
                    print('Fail sct_label_vertebrae')
                    discs_coords = np.array([]) # Fail
            
            subject_name = dir_name
            # Edit coordinates in txt file
            # line = subject_name contrast disc_num gt_coords sct_discs_coords hourglass_coords spinenet_coords
            split_lines = edit_subject_lines_txt_file(coords=discs_coords, txt_lines=split_lines, subject_name=subject_name, contrast=contrast, method_name='sct_discs_coords')

    for num in range(len(split_lines)):
        split_lines[num] = ' '.join(split_lines[num])
        
    with open(txt_file,"w") as f:
        f.writelines(split_lines)