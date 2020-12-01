# This file makes neural net modules from the command line by importing an appropriate template module
import importlib as il 
import sys

import config


def make_model_from_template(model):
    print('\n',config.template_path+model.settings.template,'\n',model.settings.template,'\n')
    sys.path.append(config.template_path+model.settings.template)
    module = il.import_module(model.settings.template) # Import the specified model from file with same name
    learned = None # TBI 
    return module, learned
    

