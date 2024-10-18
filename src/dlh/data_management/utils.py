import os
import re
import time
import json
from pathlib import Path

def get_img_path_from_label_path(str_path):
    """
    This function does 2 things: ⚠️ Files need to be stored in a BIDS compliant dataset
        - Step 1: Remove label suffix and derivative entities after the contrast (e.g. "_label-discs_dlabel"). All the information between the MRI contrast and the file extension will be removed.
        - Step 2: Remove derivatives path (e.g. derivatives/labels/). The first folders is always called derivatives but the second may vary (e.g. labels_soft)

    :param path: absolute path to the label img. Example: /<path_to_BIDS_data>/derivatives/labels/sub-amuALT/anat/sub-amuALT_T1w_labels-disc-manual.nii.gz
    :return: img path. Example: /<path_to_BIDS_data>/sub-amuALT/anat/sub-amuALT_T1w.nii.gz

    """
    # Load path
    path = Path(str_path)

    # Extract file extension
    ext = ''.join(path.suffixes)

    # Get img name
    path_list = path.name.split('_')
    suffixes_pos = [1 if len(part.split('-')) == 1 else 0 for part in path_list]
    contrast_idx = suffixes_pos.index(1) # Find the first suffix

    img_name = '_'.join(path.name.split('_')[:contrast_idx+1]) + ext
    
    # Create a list of the directories
    dir_list = str(path.parent).split('/')

    # Remove "derivatives" and "labels" folders
    derivatives_idx = dir_list.index('derivatives')
    dir_path = '/'.join(dir_list[0:derivatives_idx] + dir_list[derivatives_idx+2:])

    # Recreate img path
    img_path = os.path.join(dir_path, img_name)

    return img_path


##
def get_seg_path_from_label_path(label_path, seg_suffix='_seg'):
    """
    This function remove the label suffix to add the segmentation suffix
    """
    # Load path
    path = Path(label_path)

    # Extract file extension
    ext = ''.join(path.suffixes)

    # Find contrast index
    path_list = path.name.replace(ext, '').split('_')
    suffixes_pos = [1 if len(part.split('-')) == 1 else 0 for part in path_list]
    contrast_idx = suffixes_pos.index(1) # Find suffix

    # Get img name
    seg_name = '_'.join(path_list[:contrast_idx+1]) + seg_suffix + ext
    seg_path = path.parent / seg_name

    return str(seg_path)

##
def get_cont_path_from_other_cont(str_path, cont):
    """
    :param str_path: absolute path to the input nifti img. Example: /<path_to_BIDS_data>/sub-amuALT/anat/sub-amuALT_T1w.nii.gz
    :param cont: contrast of the target output image stored in the same data folder. Example: T2w
    :return: path to the output target image. Example: /<path_to_BIDS_data>/sub-amuALT/anat/sub-amuALT_T2w.nii.gz

    """
    # Load path
    path = Path(str_path)

    # Extract file extension
    ext = ''.join(path.suffixes)

    # Remove input contrast from name
    path_list = path.name.split(ext)[0].split('_')
    suffixes_pos = [1 if len(part.split('-')) == 1 else 0 for part in path_list]
    contrast_idx = suffixes_pos.index(1) # Find suffix

    # New image name
    img_name = '_'.join(path_list[:contrast_idx]+[cont]+path_list[contrast_idx+1:]) + ext

    # Recreate img path
    img_path = os.path.join(str(path.parent), img_name)

    return img_path


##
def fetch_subject_and_session(filename_path):
    """
    Get subject ID, session ID and filename from the input BIDS-compatible filename or file path
    The function works both on absolute file path as well as filename
    :param filename_path: input nifti filename (e.g., sub-001_ses-01_T1w.nii.gz) or file path
    (e.g., /home/user/MRI/bids/derivatives/labels/sub-001/ses-01/anat/sub-001_ses-01_T1w.nii.gz
    :return: subjectID: subject ID (e.g., sub-001)
    :return: sessionID: session ID (e.g., ses-01)
    :return: filename: nii filename (e.g., sub-001_ses-01_T1w.nii.gz)
    :return: contrast: MRI modality (dwi or anat)
    :return: echoID: echo ID (e.g., echo-1)
    :return: acquisition: acquisition (e.g., acq_sag)
    Copied from https://github.com/spinalcordtoolbox/manual-correction
    """

    _, filename = os.path.split(filename_path)              # Get just the filename (i.e., remove the path)
    subject = re.search('sub-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    subjectID = subject.group(0)[:-1] if subject else ""    # [:-1] removes the last underscore or slash

    session = re.search('ses-(.*?)[_/]', filename_path)     # [_/] means either underscore or slash
    sessionID = session.group(0)[:-1] if session else ""    # [:-1] removes the last underscore or slash

    echo = re.search('echo-(.*?)[_]', filename_path)     # [_/] means either underscore or slash
    echoID = echo.group(0)[:-1] if echo else ""    # [:-1] removes the last underscore or slash

    acq = re.search('acq-(.*?)[_]', filename_path)     # [_/] means either underscore or slash
    acquisition = acq.group(0)[:-1] if acq else ""    # [:-1] removes the last underscore or slash
    # REGEX explanation
    # . - match any character (except newline)
    # *? - match the previous element as few times as possible (zero or more times)

    contrast = 'dwi' if 'dwi' in filename_path else 'anat'  # Return contrast (dwi or anat)

    return subjectID, sessionID, filename, contrast, echoID, acquisition


def fetch_contrast(filename_path):
    '''
    Extract MRI contrast from a BIDS-compatible filename/filepath
    The function handles images only.
    :param filename_path: image file path or file name. (e.g sub-001_ses-01_T1w.nii.gz)
    '''
    return filename_path.rstrip(''.join(Path(filename_path).suffixes)).split('_')[-1]

##
def fetch_img_paths(config_data, split='TESTING'):
    # Get file paths based on split
    paths = config_data[split]
    img_paths = []
    for path in paths:
        # Check TYPE to get img_path
        if config_data['TYPE'] == 'IMAGE':
            img_path = path
        elif config_data['TYPE'] == 'LABEL':
            img_path = get_img_path_from_label_path(path)
        else:
            raise ValueError('TYPE error: The TYPE can only be "IMAGE" or "LABEL"')
        img_paths.append(img_path)
    return img_paths

##
def get_mask_path_from_img_path(img_path, suffix='_seg', derivatives_path='derivatives/labels'):
    """
    This function returns the mask path from an image path. Images need to be stored in a BIDS compliant dataset.

    :param img_path: String path to niftii image
    :param suffix: Mask suffix
    :param derivatives_path: Relative path to derivatives folder where labels are stored (e.i. 'derivatives/labels')
    Based on https://github.com/spinalcordtoolbox/disc-labeling-benchmark
    """
    # Extract information from path
    subjectID, sessionID, filename, contrast, echoID, acquisition = fetch_subject_and_session(img_path)

    # Extract file extension
    path_obj = Path(img_path)
    ext = ''.join(path_obj.suffixes)

    # Create mask name
    mask_name = path_obj.name.split('.')[0] + suffix + ext

    # Split path using "/" (TODO: check if it works for windows users)
    path_list = img_path.split('/')

    # Extract subject folder index
    sub_folder_idx = path_list.index(subjectID)

    # Reconstruct mask_path
    mask_path = os.path.join('/'.join(path_list[:sub_folder_idx]), derivatives_path, "/".join(path_list[sub_folder_idx:-1]), mask_name)
    return mask_path

##
def create_json(fname_nifti):
    """
    Create JSON sidecar with meta information
    :param fname_nifti: str: File path of the nifti image to associate with the JSON sidecar
    Based on https://github.com/spinalcordtoolbox/manual-correction
    """
    fname_json = fname_nifti.replace('.gz', '').replace('.nii', '.json')
    
    # Init new json dict
    json_dict = {'GeneratedBy': []}
    
    # Add new author with time and date
    json_dict['GeneratedBy'].append({'Author': 'Discs labeling Hourglass', 'Date': time.strftime('%Y-%m-%d %H:%M:%S')})
    with open(fname_json, 'w') as outfile: # w to overwrite the file
        json.dump(json_dict, outfile, indent=4)
        # Add last newline
        outfile.write("\n")
    print("JSON sidecar was created: {}".format(fname_json))
