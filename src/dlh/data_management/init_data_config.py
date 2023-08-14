import os
import argparse
import random
import math

from dlh.data_management.utils import get_full_path, get_img_path_from_label_path, fetch_contrast
from dlh.utils.test_utils import CONTRAST

# Determine specified contrasts
def init_data_config(args): 
    '''
    Create a JSON configuration file from a TXT file where images paths are specified
    '''
    contrasts_list = []
    config_dict = {}

    # Check that summing split ratios equals 1 else raise error
    if sum(list(args.split_ratio)) != 1.0:
        raise ValueError(f'The sum of the ratios need to be equal to 1 see {args.split_ratio}')

    # Load TXT file and extract paths
    txt_path = os.path.abspath(args.txt)
    with open(txt_path) as f:
        txt_paths = f.readlines()

    # Shuffle input TXT path
    if args.shuffle:
        txt_paths = random.shuffle(txt_paths)

    # Start loop to create data configuration
    for in_path in txt_paths:
        # Get full path
        path = get_full_path(in_path)

        # Check if path exist
        if os.path.isfile(path):
            # Extract contrast
            if args.type == 'LABEL':
                path = get_img_path_from_label_path(path)
            contrast = fetch_contrast(path)
            
            # Add contrast in contrasts_list
            if contrast not in contrasts_list:
                contrasts_list.append(contrast)
            
        else:
            txt_paths.remove(in_path)
            print(f'{in_path} was not added beacause does not exists')
        
        # Configure and append data split ratio
        nb_paths = len(txt_paths)
        idx = 0
        for pos, ratio in enumerate(args.split_ratio):
            if ratio != 0:
                split = ['TRAINING', 'VALIDATION', 'TESTING'][pos]
                config_dict[split] = [idx:nb_paths*sum(args.split_ratio[:])] 
                yml_lines.insert(idx, split)
                idx += math.ceil(nb_paths*ratio)
        
        # Append data type
        config_dict['TYPE'] = args.type

        # Append contrasts
        contrast_found = False
        for key, value in CONTRAST.items():
            if value.sort() == contrasts_list.sort():
                yml_lines.extend(['CONTRAST', '- ' + key])
                contrast_found = True
        if not contrast_found:
            print(f'ERROR: Contrasts {contrasts_list} are not handled yet')
        
        # Create YAML data configuration
        yml_path = os.path.join(txt_path.replace('.txt', '.yml'))
        with open(yml_path, 'w')as f:
            f.writelines(yml_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create config YAML from a TXT file which contains list of paths')
    
    ## Parameters
    parser.add_argument('--txt', type=str,
                        help='Path to TXT file that contains only image paths. (Required)')
    parser.add_argument('--type', type=str,
                        help='Type of paths specified. Choices "LABEL" or "IMAGE". (Required)')
    parser.add_argument('--split-ratio', type=tuple, default=(0.8, 0.1, 0.1),
                        help='Split ratio (TRAINING, VALIDATION, TESTING): Example (0.8, 0.1, 0.1) or (0, 0, 1) for TESTING only. Default=(0.8, 0.1, 0.1)')
    # parser.add_argument('--folds', type=int, default=1,
    #                     help='Number of configurations with different data folds')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Shuffle input paths. Default=True')
    
    init_data_config(parser.parse_args())