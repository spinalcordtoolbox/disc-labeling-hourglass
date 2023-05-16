#===================================================
## Authors: 
# - Reza Azad (rezazad68@gmail.com)
# - Nathan Molinier (nathan.molinier@gmail.com)
#===================================================

from shutil import copyfile
import os
import argparse


def main(args):
    '''
    Gather only relevant data for the hourglassnetwork.
    '''
    data_dir = args.datapath # The data need to use BIDS convention
    label_suffix = args.suffix_label
    img_suffix = args.suffix_img
    destination_path = args.output_folder
    contrasts = ['T1w', 'T2w']
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    ADD =  data_dir + "derivatives/labels/"
    ADD2 = data_dir

    list_dir = os.listdir(ADD)
    Total = 0
    for idx in range (len(list_dir)):
        if list_dir[idx].startswith('sub'):
            for contrast in contrasts:
                print(f"Processing subject {list_dir[idx]} with {contrast}")
                ## Copy the disc label   
                src1 = ADD + list_dir[idx] + '/anat/' + list_dir[idx] + img_suffix + '_' + contrast + label_suffix + '.nii.gz'
                dst1 = list_dir[idx] + '/' + list_dir[idx] + img_suffix + '_' + contrast + label_suffix +'.nii.gz'  
                
                ## Copy the image   
                src2 = ADD2 + list_dir[idx] + '/anat/'+list_dir[idx] + img_suffix + '_' + contrast + '.nii.gz'
                dst2 = list_dir[idx] + '/' +list_dir[idx] + img_suffix + '_' + contrast + '.nii.gz'           
                
                # Copy image and labels only if both are present in the dataset    
                if os.path.exists(src1) and os.path.exists(src2): 
                    out_path = os.path.join(destination_path, list_dir[idx])
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    copyfile(src1, os.path.join(destination_path, dst1))
                    copyfile(src2, os.path.join(destination_path, dst2))
                    Total += 1
                else:
                    if not os.path.exists(src1):
                        if not os.path.exists(src2):
                            print(f"{src1} and {src2} does not exist. Please check --suffix-label and --suffix-img. Or if {contrast} exists")
                        else:    
                            print(f"{src1} does not exist. Please check --suffix-label. Or if {contrast} exists")
                    else:    
                        print(f"{src2} does not exist. Please check --suffix-img. Or if {contrast} exists")

    print(f'Total number of {len(list_dir)} subject in the dataset\n{Total} files with both image and labels were copied')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather image files and vertebral labels in a new directory')
    
    ## Parameters
    parser.add_argument('--datapath', type=str, required=True,
                        help='Path to BIDS data')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Path out to output folder')
    parser.add_argument('--suffix-label', type=str, default='_labels-disc-manual',
                        help='Specify label suffix (default= "_labels-disc-manual")') 
    parser.add_argument('--suffix-img', type=str, default='',
                        help='Specify img suffix (default= "")') 
    
    main(parser.parse_args())