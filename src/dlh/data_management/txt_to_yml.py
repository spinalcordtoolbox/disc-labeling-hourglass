import argparse

# Determine specified contrasts




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create config YAML from a TXT file which contains list of paths')
    
    ## Parameters
    parser.add_argument('--txt', type=str,
                        help='Path to TXT file that contains only image paths. (Required)')
    parser.add_argument('--split-ratio', type=str, default=(0.8, 0.1, 0.1),
                        help='Split ratio (TRAINING, VALIDATION, TESTING): Example (0.8, 0.1, 0.1) or (0, 0, 1) for TESTING only. Default=(0.8, 0.1, 0.1)')
    parser.add_argument('--folds', type=int, default=1,
                        help='Number of configurations with different data folds')
    