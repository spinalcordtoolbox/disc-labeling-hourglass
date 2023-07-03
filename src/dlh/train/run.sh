# Script to run multiple commands

# ----------------- The script starts here --------------------#

# Select environment
source /usr/local/miniforge3/etc/profile.d/conda.sh
conda activate sct_hg_env

# Run training on hourglass
python src/dlh/train/main.py --datapath ../../data/preprocessed_data/spinegeneric_vert/ -c t2 --ndiscs 11
python src/dlh/train/main.py --datapath ../../data/preprocessed_data/spinegeneric_vert/ -c t1_t2 --ndiscs 11
python src/dlh/train/main.py --datapath ../../data/preprocessed_data/spinegeneric_vert/ -c t1 --ndiscs 11