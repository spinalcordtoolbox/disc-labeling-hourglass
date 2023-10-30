import os
import argparse
import random
import json
import itertools

from dlh.data_management.utils import get_img_path_from_label_path, fetch_contrast, fetch_subject_and_session
from dlh.utils.test_utils import CONTRAST

CONTRAST_LOOKUP = {tuple(sorted(value)): key for key, value in CONTRAST.items()}


# Determine specified contrasts
def init_data_config(args): 
    """
    Create a JSON configuration file from a TXT file where images paths are specified
    """
    if (args.split_validation + args.split_test) > 1:
        raise ValueError("The sum of the ratio between testing and validation cannot exceed 1")

    # Get input paths, could be label files or image files,
    # and make sure they all exist.
    file_paths = [os.path.abspath(path.replace('\n', '')) for path in open(args.txt)]
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
        raise ValueError("missing files:\n" + '\n'.join(missing_paths))

    # Look up the right code for the set of contrasts present
    contrasts = CONTRAST_LOOKUP[tuple(sorted(set(map(fetch_contrast, img_paths))))]

    config = {
        'TYPE': args.type,
        'CONTRASTS': contrasts,
    }

    # Split into training, validation, and testing sets
    split_ratio = (1 - (args.split_validation + args.split_test), args.split_validation, args.split_test) # TRAIN, VALIDATION, and TEST
    config_paths = label_paths if args.type == 'LABEL' else img_paths
    random.shuffle(config_paths)
    splits = [0] + [
        int(len(config_paths) * ratio)
        for ratio in itertools.accumulate(split_ratio)
    ]
    for key, (begin, end) in zip(
        ['TRAINING', 'VALIDATION', 'TESTING'],
        pairwise(splits),
    ):
        config[key] = config_paths[begin:end]

    # Save the config
    config_path = args.txt.replace('.txt', '') + '.json'
    json.dump(config, open(config_path, 'w'), indent=4)

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    # based on https://docs.python.org/3.11/library/itertools.html
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create config JSON from a TXT file which contains list of paths')
    
    ## Parameters
    parser.add_argument('--txt',
                        help='Path to TXT file that contains only image paths. (Required)')
    parser.add_argument('--type', choices=('LABEL', 'IMAGE'),
                        help='Type of paths specified. Choices "LABEL" or "IMAGE". (Required)')
    parser.add_argument('--split-validation', type=float, default=0.1,
                        help='Split ratio for validation. Default=0.1')
    parser.add_argument('--split-test', type=float, default=0.1,
                        help='Split ratio for testing. Default=0.1')
    
    args = parser.parse_args()
    
    if args.split_test > 0.9:
        args.split_validation = 1 - args.split_test
    
    init_data_config(parser.parse_args())
