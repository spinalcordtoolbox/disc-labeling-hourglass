import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

from dlh.models.hourglass import hg
from dlh.models.atthourglass import atthg
from dlh.utils.train_utils import image_Dataset
from dlh.utils.image import Image, zeros_like
from dlh.utils.config2parser import config2parser
from dlh.utils.test_utils import CONTRAST, extract_skeleton, load_img_only, fetch_img_paths, get_mask_path_from_img_path
from dlh.data_management.utils import fetch_subject_and_session

#---------------------------Test Hourglass Network----------------------------
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get hourglass training parameters
    config_hg = config2parser(args.config_hg)
    train_contrast = config_hg.train_contrast
    ndiscs = config_hg.ndiscs
    att = config_hg.att
    stacks = config_hg.stacks
    blocks = config_hg.blocks
    skeleton_dir = config_hg.skeleton_folder
    weights_dir = config_hg.weight_folder

    # Read json file and create a dictionary
    with open(args.config_data, "r") as file:
        config_data = json.load(file)
    
    # Fetch contrast info from config data
    data_contrast = CONTRAST[config_data['CONTRASTS']] # contrast_str is a unique string representing all the contrasts
    
    # Error if data contrast not in training
    for cont in data_contrast:
        if cont not in CONTRAST[train_contrast] and args.trained_contrast_only:
            raise ValueError(f"Data contrast {cont} not used for training.")
    
    # Load images
    print('loading images...')
    imgs_test, subjects_test, original_shapes = load_img_only(config_data=config_data,
                                                              split='TESTING')
    
    img_paths = fetch_img_paths(config_data=config_data,
                                split='TESTING')

    # Verify if skeleton exists before running test
    path_skeleton = os.path.join(skeleton_dir, f'{train_contrast}_Skelet_ndiscs_{ndiscs}.npy')
    if os.path.exists(path_skeleton):
        print(f'Processing with hourglass trained on contrast {train_contrast}')
        norm_mean_skeleton = np.load(path_skeleton)
        
        # Load network weights
        if att:
            model = atthg(num_stacks=stacks, num_blocks=blocks, num_classes=ndiscs)
        else:
            model = hg(num_stacks=stacks, num_blocks=blocks, num_classes=ndiscs)

        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(os.path.join(weights_dir, f'model_{train_contrast}_att_stacks_{stacks}_ndiscs_{ndiscs}'), map_location='cpu')['model_weights'])

        # Create Dataloader
        full_dataset_test = image_Dataset(images=imgs_test, 
                                        subjects_names=subjects_test,
                                        num_channel=ndiscs,
                                        use_flip = False,
                                        load_mode='test'
                                        ) 
        
        MRI_test_loader   = DataLoader(full_dataset_test, 
                                    batch_size= 1, 
                                    shuffle=False, 
                                    num_workers=0
                                    )
        model.eval()
        
        # Extract discs coordinates from the test set
        for i, (inputs, subject_name) in enumerate(MRI_test_loader):
            print(f'Running inference on {subject_name[0]}')

            inputs = inputs.to(device)
            output = model(inputs) 
            output = output[-1]
            
            # Fetch contrast, subject, session and echo
            img_path = img_paths[i]
            subjectID, sessionID, filename, _, echoID, acq = fetch_subject_and_session(img_path)
            
            print('Extracting skeleton')
            try:
                prediction, pred_discs_coords = extract_skeleton(inputs=inputs, outputs=output, norm_mean_skeleton=norm_mean_skeleton, Flag_save=True)
                
                # Convert pred_discs_coords to original image size
                pred_shape = prediction[0,0].shape 
                original_shape = original_shapes[i] 
                pred = np.array([[(round(coord[0])/pred_shape[0])*original_shape[0], (round(coord[1])/pred_shape[1])*original_shape[1], int(disc_num)] for disc_num, coord in pred_discs_coords[0].items()]).astype(int)
                
                # Create empty mask
                img = Image(img_path).change_orientation('RSP')
                out_mask = zeros_like(img)

                # Get transpose
                pred_t = np.rint(np.transpose(pred)).astype(int)

                # Extract middle slice of the image (the hourglass is a 2D method)
                middle_slice = np.array([img.data.shape[0]//2]*pred.shape[0])

                # Add discs coords to mask
                out_mask.data[middle_slice, pred_t[0], pred_t[1]] = pred_t[2]

                # Save prediction mask
                out_path = get_mask_path_from_img_path(img_path, suffix='_label-discs', derivatives_path='/derivatives/labels')
                out_mask.save(path=out_path, dtype='float32')
            
            except ValueError:
                print(f'Failed detection with filename {filename}')

    else:
        raise ValueError(f'Path to skeleton {path_skeleton} does not exist'
                f'Please check if contrasts {train_contrast} was used for training')     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Hourglass Network coordinates to text file')

    ## Parameters
    # All mandatory parameters                         
    parser.add_argument('--config-data', type=str, metavar='<folder>', required=True,
                        help='Config JSON file where every label/image used for TESTING has its path specified ~/<your_path>/config_data.json (Required)')                               
    parser.add_argument('--config-hg', type=str, required=True,
                        help='Config file where hourglass training parameters are stored Example: Example: ~/<your_path>/config.json (Required)')  # Hourglass config file
    parser.add_argument('--trained-contrast-only', type=bool, default=False,
                        help='If True this script will generate an error when the input contrast is different from those used during the training (default=False)')
    
    
    # Run Hourglass Network on input data
    run_inference(parser.parse_args())

    print('Hourglass inference has been computed')