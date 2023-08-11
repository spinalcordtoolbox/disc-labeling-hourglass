import os
import re
import sys
import yaml

def fetch_yaml_config(config_file):
    """
    Fetch configuration from YAML file
    :param config_file: YAML file
    :return: yaml dictionary
    Based on https://github.com/spinalcordtoolbox/manual-correction
    """
    config_file = os.path.abspath(config_file)
    # Check if file exist
    if os.path.isfile(config_file):
        fname_yml = config_file
    else:
        sys.exit("ERROR: Input yml file {} does not exist or path is wrong.".format(config_file))
    
    # Fetch dict from yml
    with open(fname_yml, 'r') as stream:
        try:
            dict_yml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return dict_yml

def get_img_path_from_label_path(path):
    """
    This function does 2 things: ⚠️ Files need to be stored in a BIDS compliant dataset
        - Step 1: Remove label suffix (e.g. "_labels-disc-manual"). The suffix is always between the MRI contrast and the file extension.
        - Step 2: Remove derivatives path (e.g. derivatives/labels/). The first folders is always called derivatives but the second may vary (e.g. labels_soft)

    :param path: absolute path to the label img. Example: /<path_to_BIDS_data>/derivatives/labels/sub-amuALT/anat/sub-amuALT_T1w_labels-disc-manual.nii.gz
    :return: img path. Example: /<path_to_BIDS_data>/sub-amuALT/anat/sub-amuALT_T1w.nii.gz

    """
    # Extract file extension
    ext = '.' + '.'.join(path.split('.')[1:])

    # Get img name
    img_name = '_'.join(os.path.basename(path).split('_')[:-1])
    
    # Create a list of the directories
    dir_list = path.split('/')

    # Remove "derivatives" and "labels" folders
    derivatives_idx = dir_list.index('derivatives')
    dir_path = '/' + '/'.join(dir_list[0:derivatives_idx] + dir_list[derivatives_idx+2:])

    # Recreate img path
    img_path = dir_path + img_name + ext

    return img_path

def fetch_subject_and_session(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subjectID: subject ID (e.g., sub-001)
    :return: sessionID: session ID (e.g., ses-01)
    :return: filename: nii filename (e.g., sub-001_ses-01_T1w.nii.gz)
    Copied from https://github.com/spinalcordtoolbox/manual-correction
    """

    _, filename = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    subject = re.search('sub-(.*?)[_/]', filename_path)
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash
    session = re.findall(r'ses-..', filename_path)
    sessionID = session[0] if session else ""               # Return None if there is no session
    contrast = 'dwi' if 'dwi' in filename_path else 'anat'  # Return contrast (dwi or anat)
    # REGEX explanation
    # \d - digit
    # \d? - no or one occurrence of digit
    # *? - match the previous element as few times as possible (zero or more times)

    return subjectID, sessionID, filename, contrast