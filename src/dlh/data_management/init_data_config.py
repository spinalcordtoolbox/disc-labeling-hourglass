import os
import argparse
import random
import math
import itertools

from dlh.data_management.utils import get_full_path, get_img_path_from_label_path, fetch_contrast
from dlh.utils.test_utils import CONTRAST

CONTRAST_LOOKUP = {tuple(sorted(value)): key for key, value in CONTRAST.items()}


# Determine specified contrasts
def init_data_config(args): 
    """
    Create a JSON configuration file from a TXT file where images paths are specified
    """
    if sum(args.split_ratio) != 1:
        raise ValueError("split ratios don't add up exactly to 1")

    # Get input paths, could be label files or image files,
    # and make sure they all exist.
    file_paths = [os.path.abspath(path) for path in open(args.txt).readlines()]
    if args.type == 'LABEL':
        label_paths = file_paths
        img_paths = [get_img_path_from_label_path(lp) for lp in label_paths]
        file_paths = label_paths + img_paths
    elif args.type == 'IMAGE':
        img_paths = file_paths
    else:
        raise ValueError(f"invalid args.type: {args.type}")
    missing_paths = [
        path for path in file_paths
        if not os.path.isfile(path)
    ]
    if missing_paths:
        raise ValueError(f"missing files:\n{'\n'.join(missing_paths)}")

    # Look up the right code for the set of contrasts present
    contrasts = CONTRAST_LOOKUP[tuple(sorted(set(map(fetch_contrast, img_paths))))]

    config = {
        'TYPE': args.type,
        'CONTRASTS': contrasts,
    }

    # Split into training, validation, and testing sets
    config_paths = label_paths if args.type == 'LABEL' else img_paths
    random.shuffle(config_paths)
    splits = [0] + [
        int(len(config_paths)) * ratio
        for ratio in itertools.accumulate(args.split_ratio)
    ]
    for key, (begin, end) in zip(
        ['TRAINING', 'VALIDATION', 'TESTING'],
        itertools.pairwise(splits),
    ):
        config[key] = config_paths[begin:end]

    # Save the config
    config_path = args.txt.removesuffix('.txt') + '.json'
    json.dump(config, open(config_path, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create config YAML from a TXT file which contains list of paths')
    
    ## Parameters
    parser.add_argument('--txt',
                        help='Path to TXT file that contains only image paths. (Required)')
    parser.add_argument('--type', choices=('LABEL', 'IMAGE'),
                        help='Type of paths specified. Choices "LABEL" or "IMAGE". (Required)')
    parser.add_argument('--split-ratio', type=tuple, default=(0.8, 0.1, 0.1),
                        help='Split ratio (TRAINING, VALIDATION, TESTING): Example (0.8, 0.1, 0.1) or (0, 0, 1) for TESTING only. Default=(0.8, 0.1, 0.1)')
    
    init_data_config(parser.parse_args())
