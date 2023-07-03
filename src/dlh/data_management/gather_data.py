#===================================================
## Authors: 
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
# From https://github.com/spinalcordtoolbox/disc-labeling-benchmark 
#===================================================

from shutil import copyfile
import os
import argparse


def main(args):
    '''
    Gather only relevant data for the hourglass.
    '''
    bids_dir = args.datapath # The data need to use BIDS convention
    label_disc_suffix = args.suffix_label_disc
    seg_suffix = args.suffix_seg
    img_suffix = args.suffix_img
    destination_path = args.output_folder
    contrasts = ['T1w', 'T2w']
    derivatives_dir =  os.path.join(bids_dir,"derivatives/labels/")

    # Create output folder if does not exist
    if not os.path.exists(destination_path):
        print(f'Output folder {destination_path} was created')
        os.makedirs(destination_path)
    
    list_dir = os.listdir(bids_dir)
    Total = 0
    for sub in list_dir:
        if sub.startswith('sub'):
            if 'anat' not in os.listdir(os.path.join(bids_dir, sub)): # Check if sessions are provided in BIDS
                sessions = os.listdir(os.path.join(bids_dir, sub))
            else:
                sessions = ['']
            for ses in sessions:
                for contrast in contrasts:
                    if ses != '':
                        print(f"Processing subject {sub} during session {ses} with {contrast} contrast")
                    else:
                        print(f"Processing subject {sub} with {contrast} contrast")
                    
                    ## Copy the image
                    src1, img1 = create_path(bids_folder=bids_dir, sub=sub, ses=ses, img_suffix=img_suffix, contrast=contrast, label_suffix='', ext='.nii.gz')
                    dst1 = os.path.join(destination_path, sub, img1)
                    
                    ## Copy the disc label   
                    src2, img2 = create_path(bids_folder=derivatives_dir, sub=sub, ses=ses, img_suffix=img_suffix, contrast=contrast, label_suffix=label_disc_suffix, ext='.nii.gz')
                    dst2 = os.path.join(destination_path, sub, img2)  
                    
                    ## Copy the segmentation
                    src3, img3 = create_path(bids_folder=derivatives_dir, sub=sub, ses=ses, img_suffix=img_suffix, contrast=contrast, label_suffix=seg_suffix, ext='.nii.gz')
                    dst3 = os.path.join(destination_path, sub, img3)
                    
                    # Copy image and discs labels if both are present in the dataset or if args.img_only is True   
                    if os.path.exists(src1) and os.path.exists(src2) or os.path.exists(src1) and args.img_only: 
                        out_path = os.path.join(destination_path, sub)
                        if not os.path.exists(out_path):
                            os.makedirs(out_path)
                        if os.path.exists(src1):
                            copyfile(src1, dst1)
                        if os.path.exists(src2):
                            copyfile(src2, dst2)
                        if os.path.exists(src3):
                            copyfile(src3, dst3)
                        Total += 1
                    else:
                        if not os.path.exists(src1):
                            if not os.path.exists(src2):
                                print(f"{src1} and {src2} does not exist. Please check --suffix-label and --suffix-img. Or if {contrast} exists")
                            else:    
                                print(f"{src1} does not exist. Please check --suffix-label. Or if {contrast} exists")
                        else:    
                            print(f"{src2} does not exist. Please check --suffix-img. Or if {contrast} exists")

    print(f'Total number of {len(derivatives_dir)} subject in the dataset\n{Total} subjects were computed')


def create_path(bids_folder, sub, ses='', img_suffix='', contrast='', label_suffix='', ext='.nii.gz'):
    '''
    This function creates the full path to an image and the image name only
    :param bids_folder: full path to BIDS directory
    :param sub: subject name
    :param ses: session id
    :param img_suffix: other additional suffix after session
    :param contrast: MRI contrast
    :param label_suffix: method suffix for processed images (i.e.: '_seg', '_label_discs')
    :param ext: image extension
    '''
    img_path = os.path.join(bids_folder, sub, ses, 'anat')
    img_name = sub
    for info in [ses, img_suffix, contrast, label_suffix]:
        if info != '':
            if info.startswith('_'):
                img_name += info
            else:
                img_name += '_' + info
    return os.path.join(img_path, img_name + ext), img_name + ext

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather image files and vertebral labels in a new directory')
    
    ## Parameters
    parser.add_argument('--datapath', type=str, required=True,
                        help='Path to BIDS data')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Path out to output folder')
    parser.add_argument('--img-only', type=bool, default=False,
                        help='If True all the images will be moved even if discs labels does not exist'
                        ' If False images and labels will be moved only if both exist')
    parser.add_argument('--suffix-img', type=str, default='',
                        help='Specify img suffix example: sub-296085(SESSION)(IMG_SUFFIX)_T2w.nii.gz (default= "")')
    parser.add_argument('--suffix-label-disc', type=str, default='_labels-disc-manual',
                        help='Specify disc label suffix example: sub-296085(IMG_SUFFIX)_T2w(LABEL_SUFFIX).nii.gz (default= "_labels-disc-manual")') 
    parser.add_argument('--suffix-seg', type=str, default='_seg-manual',
                        help='Specify segmentation label suffix example: sub-296085(IMG_SUFFIX)_T2w(LABEL_SUFFIX).nii.gz (default= "_seg-manual")')  
    
    main(parser.parse_args())