import os
import json
import argparse

def config2parser(config_path):
    '''
    Create a parser object from a json file 
    '''
    # Read json file and create a dictionary
    with open(config_path, "r") as file:
        config_dict = json.load(file)

    return argparse.Namespace(**config_dict)


def parser2config(args, path_out):
    '''
    Extract the parameters from an input parser to create a config json file
    :param args: parser arguments
    :param path_out: path out of the config file
    '''
    # Check if path_out exists or create it
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # Serializing json
    json_object = json.dumps(vars(args), indent=4)
    
    # Writing to sample.json
    json_path = os.path.join(os.path.abspath(path_out),f'config_{os.path.basename(args.datapath)}_{args.contrasts}.json')
    with open(json_path, "w") as outfile:
        print(f"The config file {json_path} with all the training parameters was created")
        outfile.write(json_object)
    