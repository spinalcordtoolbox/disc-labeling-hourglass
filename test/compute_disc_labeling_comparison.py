import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spinalcordtoolbox.image import Image
import csv
import pandas as pd

from dlh.utils.test_utils import CONTRAST, visualize_discs, edit_metric_csv

def compare_methods(args):
    if args.datapath != None:
        datapath = args.datapath
    contrast = CONTRAST[args.modality][0]
    txt_file_path = args.input_txt_file
    output_folder = os.path.join(args.output_folder, f'out_{contrast}')
    computed_methods = args.computed_methods
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # Load disc_coords txt file
    with open(txt_file_path,"r") as f:  # Checking already processed subjects from txt file
        file_lines = f.readlines()
        split_lines = [line.split(' ') for line in file_lines]
    
    # Extract processed subjects --> subjects with a ground truth
    # line = subject_name contrast disc_num ground_truth_coord sct_label_vertebrae_coord hourglass_coord spinenet_coord
    processed_subjects = []
    for line in split_lines[1:]:
        if (line[0] not in processed_subjects) and (line[1]==contrast) and (line[7]!='None'): # 3rd colum corresponds to ground truth coordinates
            processed_subjects.append(line[0])
    
    # Initialize metrics
    methods_results = {}
    
    nb_subjects = len(processed_subjects)
    for method in computed_methods:
        print(f"Computing method {method}")
        for subject in processed_subjects:
            methods_results, pred_discs_list = edit_metric_csv(methods_results, txt_lines=split_lines, subject_name=subject, contrast=contrast, method_name=method, nb_subjects=nb_subjects)
            
            # Visualize discs on image
            if args.datapath != None:
                img_3D = Image(os.path.join(datapath, subject, f'{subject}_{contrast}.nii.gz')).change_orientation('RSP').data
                shape = img_3D.shape
                img_2D = img_3D[shape[0]//2, :, :]
                if pred_discs_list.size != 0: # Check if not empty
                    visualize_discs(input_img=img_2D, coords_list=pred_discs_list, out_path=os.path.join(output_folder, f'{subject}_{method}.png'))
            
    
    if args.create_csv:
        # Get fields for csv conversion    
        fields = ['subject'] + [key for key in methods_results[subject].keys()]
        
        csv_path = txt_file_path.replace('discs_coords.txt', 'computed_metrics.csv')
        with open(csv_path, "w") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            for k,d in sorted(methods_results.items()):
                w.writerow(mergedict({'subject': k},d))
    
    methods_short = [method.split('_coords')[0] for method in computed_methods] # Remove '_coords' suffix
    save_graphs(output_folder, methods_results, methods_short)
    return

def mergedict(a,b):
    a.update(b)
    return a

def save_graphs(output_folder, methods_results, methods_list):
    # Isolate total dict 
    dict_total = methods_results['total']
    del methods_results['total']
            
    # Extract subjects and metrics
    subjects, subject_metrics = np.array(list(methods_results.keys())), list(methods_results.values())
    metrics_name = np.array(list(subject_metrics[0].keys()))
    metrics_values = np.array([list(sub_metrics.values()) for sub_metrics in subject_metrics])
    
    # Plot total graph
    # L2 error
    l2_error = [metrics_values[:,np.where(metrics_name == f'l2_mean_{method}')[0][0]] for method in methods_list]
    l2_mean = [dict_total[f'l2_mean_{method}'] for method in methods_list]
    l2_std = [dict_total[f'l2_std_{method}'] for method in methods_list]
    out_path = os.path.join(output_folder,'l2_error.png')
    #save_bar(methods=methods_list, mean=l2_mean, std=l2_std, output_path=out_path, x_axis='Methods', y_axis='L2_error (pixels)')
    save_violin(methods=methods_list, values=l2_error, output_path=out_path, x_axis='Methods', y_axis='L2_error (pixels)')
    
    # # Save total Dice score DSC
    # DSC_hg = dict_total['DSC_hg']
    # DSC_sct = dict_total['DSC_sct']
    # DSC_spn = dict_total['DSC_spn']
    # out_path = os.path.join(output_folder,'labels_dice_score.png')
    # save_bar(x='Total',
    #          y1=DSC_hg,
    #          y2=DSC_sct,
    #          y3=DSC_spn,
    #          output_path=out_path,
    #          x_axis='Total', 
    #          y_axis='Labels accuracy using DSC', 
    #          label1 ='hourglass_network', 
    #          label2 ='sct_label_vertebrae', 
    #          label3 ='spinenetv2_label_vertebrae'
    #          )

    
def save_bar(methods, mean, std, output_path, x_axis='Subjects', y_axis='L2_error (pixels)'):
    '''
    Create a bar graph
    :param methods: String list of the methods name
    :param mean: List of the mean corresponding to the methods name
    :param str: List of the str corresponding to the methods name
    :param output_path: Path to output folder where figures will be stored
    :param x_axis: x-axis name
    :param y_axis: y-axis name
    '''
    
    # set width of bar
    barWidth = 0.25
    plt.figure(figsize =(len(methods), 10))
    #plt.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.95)
        
    # Set position of bar on X axis
    br1 = np.arange(len(methods))
    
    # Make the plot 
    fig, ax = plt.subplots()      
    ax.bar(br1, mean, yerr=std, align='center', color='b', width = barWidth, edgecolor ='grey')
    plt.title(y_axis, fontweight ='bold', fontsize = 20)
    
    # Create axis and adding Xticks
    plt.xlabel(x_axis, fontweight ='bold', fontsize = 15)
    plt.ylabel(y_axis, fontweight ='bold', fontsize = 15)
    plt.xticks([r for r in range(len(methods))], methods)
    
    # Save plot
    plt.legend()
    plt.savefig(output_path)


def save_violin(methods, values, output_path, x_axis='Subjects', y_axis='L2_error (pixels)'):
    '''
    Create a bar graph
    :param methods: String list of the methods name
    :param values: Values associated with the methods
    :param output_path: Path to output folder where figures will be stored
    :param x_axis: x-axis name
    :param y_axis: y-axis name
    '''
    
    # set width of bar
        
    # Set position of bar on X axis
    result_dict = {}
    for i, method in enumerate(methods):
        result_dict[method]=values[i]
    result_df = pd.DataFrame(data=result_dict)

    # Make the plot 
    plot = sns.violinplot(data=result_df)  
    
    # Create axis and adding Xticks
    
    # Save plot
    plot.figure.savefig(output_path)

 
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute metrics on sct and hourglass disc estimation')
    
    parser.add_argument('--datapath', type=str, metavar='N', default=None,
                        help='Path to dataset cf gather_data format')
    parser.add_argument('-txt', '--input-txt-file', type=str, metavar='N', required=True,
                        help='Input txt file with the methods coordinates')
    parser.add_argument('-o', '--output-folder', type=str, metavar='N', required=True,
                        help='Output folder for created graphs') 
    parser.add_argument('-c', '--modality', type=str, metavar='N', required=True,
                        help='Data modality')
    parser.add_argument('-csv', '--create-csv', metavar='N', default=True,
                        help='Output csv file with computed metrics') 
    parser.add_argument('--computed-methods', 
                        metavar='N', 
                        default=['sct_discs_coords', 
                                 'spinenet_coords', 
                                 'hourglass_t1_coords', 
                                 'hourglass_t2_coords', 
                                 'hourglass_t1_t2_coords'],
                        help='Methods on which metrics will be computed') 
    
    
    compare_methods(parser.parse_args())