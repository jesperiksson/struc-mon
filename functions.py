

# Imports
# Standard stuff
import os
import pandas as pd
# Self made stuff
import config
import Databatch
from Model import Model, NeuralNet, TimeSeriesNeuralNet
from Settings import *

### FUNCTIONS FOR CREATING OR LOADING A MODEL ###
def new_model():
    settings = get_settings() #lives in a separate file since it is large
    #save_settings(settings)
    data = Series_Stack(settings, 'new',config.file_path)
    model = TimeSeriesNeuralNet(settings, existing_model = False)
    return settings, data, model

def load_model(name = None):
    if name == None:
        settings = load_settings(input('Which model?'))
    else:
        settings = load_settings(name)
        data = Series_Stack(settings, 'old')
    neural_net = NeuralNet(settings, existing_model = True)
    return settings, data, neural_net

# Settings, maybe make a class later

def save_settings(settings):
    with open(settings['file_path'] + settings['name'] +'.pkl', 'wb') as f:
        pickle.dump(arch, f, pickle.HIGHEST_PROTOCOL)

def load_settings(settings):
    with open(file_path + arch['name'] +'.pkl', 'rb') as f:
        return pickle.load(f)



### FUNCTIONS FOR CONTINUOSLY UPDATING ###

def continuosly_update(model):
    # Start the scheduler
    scheduler = BackgroundScheduler()
    #scheduler.daemonic = False
    scheduler.start()
    scheduler.add_job(check_for_new_files(model), 'interval', minutes = interval) 
    return

def check_for_new_files(model):
    to_learn = scan(model.settings[learned])
    if model.settings['data_function'] == 'one_by_one':
        for i in range(len(to_learn)):
            data, learned = get_data_one_by_one(to_learn[i])
            model.settings.update({'learned' : learned})
            NeuralNet.train(model, data)
    elif model.settings['data_function'] == 'all_at_once':
        data = get_data_all_at_once(learned = {})
        model.settings.update({'learned' : learned})
        NeuralNet.train(model, data)
    save_model(model)
    return

### FUNCTIONS FOR ANALYZING ###

def analyze(model):
    pass
