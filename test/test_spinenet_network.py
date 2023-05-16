import os
import matplotlib.pyplot as plt
import argparse
from matplotlib.patches import Polygon, Circle 
import numpy as np
from scipy.ndimage import zoom

from spinalcordtoolbox.image import Image, get_dimension
from spinenet import SpineNet, io

from dlh.utils.test_utils import CONTRAST, VERT_DISC, swap_y_origin, coord2list, project_on_spinal_cord, edit_subject_lines_txt_file 

#---------------------------Test spinenet--------------------------
def test_spinenet(args, test_mode=False):
    '''
    Use spinenet to find the vertebrae discs coordinates and append them
    to a txt file
    '''
    
    datapath = os.path.abspath(args.datapath)
    contrast = CONTRAST[args.contrast][0]
    img_suffix = args.suffix_img
    
    # load in spinenet
    spnt = SpineNet(device='cuda:0', verbose=True, scan_type='whole')
    
    # Extract txt file lines
    if not test_mode:
        txt_file = args.out_txt_file
        prefix = 'sub'
        with open(txt_file,"r") as f:
            file_lines = f.readlines()
            split_lines = [line.split(' ') for line in file_lines]
    else:
        prefix = args.sub
        
    print('Processing with spinenet')
    for dir_name in os.listdir(datapath):
        if dir_name.startswith(prefix):
            subject_name = dir_name
            file_name = subject_name + img_suffix + '_' + contrast + '.nii.gz'
            img_path = os.path.join(datapath, dir_name, file_name)  # path to the original image
            
            # img_niftii --> 3D image: shape = (64, 320, 320)
            img_niftii = Image(img_path).change_orientation("RSP")
            nx, ny, nz, nt, px, py, pz, pt = get_dimension(img_niftii)
            pixel_spacing = np.array([py, pz])
            nb_slice = 12 # Use less slices
            if abs(px-py) < 0.2 or abs(px-pz) < 0.2: # True if isotropic according to right-left direction
                skip_slices = 4
                nx = nx//skip_slices
                slice_thickness = px*skip_slices
                #img = np.moveaxis(img_niftii.data, 0, -1)[:, :, nx//2-(nb_slice*skip_slices)//2:nx//2+(skip_slices*nb_slice)//2:skip_slices]
                img = np.moveaxis(zoom(img_niftii.data, (1/skip_slices, 1, 1)), 0, -1)[:, :, nx//2-nb_slice//2:nx//2+nb_slice//2] # Use zoom function to down sample the image to a more non-isotropic image
            else:
                slice_thickness = px
                img = np.moveaxis(img_niftii.data, 0, -1)[:, :, nx//2-nb_slice//2:nx//2+nb_slice//2]
            scan = io.SpinalScan(img, pixel_spacing, slice_thickness)
            #img = np.moveaxis(img_niftii.data, 0, -1)[:, :, nx//2-nb_slice//2:nx//2+nb_slice//2]

            # detect and identify vertebrae in scan. Note that pixel spacing information is required 
            # so SpineNet knows what size to split patches into.
            try:
                vert_dicts_niftii = spnt.detect_vb(img, scan.pixel_spacing[0])
            except RuntimeError:
                vert_dicts_niftii = 'Fail'

            if vert_dicts_niftii != 'Fail':
                # extracting discs coordinates from vertebrae detections
                discs_coords = {}
                for vert_dict in vert_dicts_niftii:
                    vert = vert_dict['predicted_label']
                    poly_sum = np.zeros_like(vert_dict['polys'][0])
                    for idx in range(len(vert_dict['polys'])):
                        poly_sum += np.array(vert_dict['polys'][idx])
                    poly_mean = poly_sum/len(vert_dict['polys'])
                    
                    top_disc = VERT_DISC[vert]
                    if top_disc in discs_coords:
                        discs_coords[top_disc] = (discs_coords[top_disc] + poly_mean[-1,:])/2 # To improve the accuracy of the positioning, we calculate the average coordinate between the top and the bottom vertebrae
                    else:
                        discs_coords[top_disc] = poly_mean[-1,:] # Extract the coordinates of the top left corner which is the closest to the top disc (due to RPI orientation)
                    
                    bottom_disc = VERT_DISC[vert]+1
                    if bottom_disc in discs_coords:
                        discs_coords[bottom_disc] = (discs_coords[bottom_disc] + poly_mean[-2,:])/2 # To improve the accuracy of the positioning, we calculate the average coordinate between the top and the bottom vertebrae
                    else:
                        discs_coords[bottom_disc] = poly_mean[-2,:] # Extract the coordinates of the bottom left corner which is the closest to the bottom disc (due to RPI orientation)
                
                # Convert discs num and coords to numpy array
                discs_num, coords = np.transpose([np.array(list(discs_coords.keys()))]), np.array(list(discs_coords.values()))
                
                # Swap to coords convention [x, y] <--> [lines(y), columns(x)]
                coords = coord2list(coords=coords)
                
                # Concatenate discs num and coords
                coords = np.concatenate((coords, discs_num), axis=1)
                
                # Project on spinalcord for 2D comparison
                seg_path = os.path.join(datapath, subject_name, f'{subject_name}_{contrast}_seg.nii.gz' )
                coords = project_on_spinal_cord(coords=coords, seg_path=seg_path, disc_num=True, proj_2d=True)
                
                # Move y origin to the bottom of the image like Niftii convention
                coords = swap_y_origin(coords=coords, img_shape=img[:,:,0].shape, y_pos=0).astype(int)
            else:
                coords = np.array([]) # Fail

            if not test_mode: 
                # Write coordinates in txt file
                # line = subject_name contrast disc_num gt_coords sct_discs_coords hourglass_coords spinenet_coords
                split_lines = edit_subject_lines_txt_file(coords=coords, txt_lines=split_lines, subject_name=subject_name, contrast=contrast, method_name='spinenet_coords')

    if not test_mode:
        for num in range(len(split_lines)):
            split_lines[num] = ' '.join(split_lines[num])
            
        with open(txt_file,"w") as f:
            f.writelines(split_lines)
    else:            
        return nb_slice, img, discs_coords, vert_dicts_niftii

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run test on spinenet')

    parser.add_argument('--datapath', default="/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/data/preprocessed_data/vertebral_data", type=str,
                        help='SCT dataset path')                               
    parser.add_argument('-c', '--contrast', default='t2', type=str, metavar='N',
                        help='MRI contrast')
    parser.add_argument('-sub', default= 'sub-beijingGE01',
                        type=str, metavar='N',help='Generated txt file') # 'sub-juntendo750w06'
    
    parser.add_argument('--suffix-label', type=str, default='_labels-disc-manual',
                        help='Specify label suffix (default= "_labels-disc-manual")') 
    parser.add_argument('--suffix-img', type=str, default='',
                        help='Specify img suffix (default= "")')
    
    nb_slice, img, discs_coords, vert_dicts_niftii = test_spinenet(parser.parse_args(), test_mode=True)
    fig = plt.figure(figsize=(40,40))
    for slice_idx in range(nb_slice):
        ax = fig.add_subplot(4,4,slice_idx+1)
        ax.imshow(img[:,:,slice_idx], cmap='gray')
        ax.set_title(f'Slice {slice_idx+1}', fontsize=60)
        ax.axis('off')
        for disc, coord in discs_coords.items():
            ax.add_patch(Circle(coord, radius=1, ec='r'))
            ax.text(coord[0]-15, coord[1], disc, color='r', fontsize=15)
        for vert_dict in vert_dicts_niftii:
            if slice_idx in vert_dict['slice_nos']:
                poly_idx = int(vert_dict['slice_nos'].index(slice_idx))
                poly = np.array(vert_dict['polys'][poly_idx])
                ax.add_patch(Polygon(poly, ec='y',fc='none'))
                ax.text(np.mean(poly[:,0]), np.mean(poly[:,1]), vert_dict['predicted_label'],c='y', ha='center',va='center', fontsize=15)

    fig.suptitle('Detected Vertebrae (all slices)', fontsize=100)
    fig.savefig('test/visualize/test_spinenet.png')