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
    if not os.path.exists(os.path.dirname(path_out)):
        os.makedirs(os.path.dirname(path_out))

    # Serializing json
    json_object = json.dumps(vars(args), indent=4)
    
    # Writing to sample.json
    with open(path_out, "w") as outfile:
        print(f"The config file {path_out} with all the training parameters was created")
        outfile.write(json_object)
    