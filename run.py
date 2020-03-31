# Modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.io import loadmat
#from datetime import datetime, date, time

# Class files
''' Imported when needed
from LSTM import * 
from MLP import *
from LSTMbatch import *
'''
# Utility file
from util import *

if __name__ == "__main__":


    '''
    Hyperparameters
    '''
    # Model
    architecture = {
        'prediction' : 'end_of_series', # end_of_series or entire_series
        'model_type' : 'MLP', # MLP, LSTM or AutoencoderLSTM
        'direction' : 'uni', # uni or bi (uni a.k.a. vanilla net)
        'n_layers' : 1,
        'n_units' : [100, 64],
        'sensor_list' : ['half', 'quarter', 'third'],
        'n_pred_units' : 50, # Number of units to be predicted
        'target' : 'E', # E or accelerations
# Metadata
        'elements' : [10, 45, 68, 90, 112, 135, 170],
        'healthy' : [33, 43, 52 , 62, 71, 81, 90, 100],
        'sensors' : [10, 45, 90, 135, 170],
        'speeds' : 20,
        'data_split' : [60, 20, 20],
# MLP-specific settings
        'delta' : 3,
        'n_pattern_steps' : 200,
        'n_target_steps' : 1,
        'MLPactivation' : 'relu'
# LSTM-specific settings
    }
    early_stopping = True
    # Data
    train_percentage = 60
    test_percentage = 20
    validation_percentage = 20
    data_split = [train_percentage, test_percentage, validation_percentage]
    assert sum(data_split) == 100
    split_mode = 'last' # or shuffle
    elements = [10, 45, 68, 90, 112, 135, 170]
    healthy = [33, 43, 52 , 62, 71, 81, 90, 100]
    sensors = [10, 45, 90, 135, 170]
       

    pred_sensor = sensor_dict[architecture['sensor_key']] # Which sensor to be the target, the other sensors are patterns
    n_sensors = len(architecture['sensor_list'])


    # Training
    epochs = 20000

    # Plotting
    do_plot_loss= True  

    # Testing
    mse_threshold = 0.000015  

    name = (architecture['prediction']+'_'+ \
           architecture['sensor_key']+'_'+ \
           architecture['model_type']+'_'+ \
           str(architecture['n_layers'])+'_'+ \
           str(architecture['n_pred_units']))

    if architecture['model_type'] == 'MLP':
        from MLP import *
    elif architecture['model_type'] == 'LSTM':
        from LSTM import *
    elif architecture['model_type'] == 'AELSTM':
        from AELSTM import *
    try:
        f = open('models/'+name+'.json')
        machine_stack = {
            name : NeuralNet(architecture,
                 data_split,
                 name,
                 pred_sensor,
                 n_sensors,
                 early_stopping,
                 existing_model = True)
        }
    except IOError:    
        machine_stack = {
            name : NeuralNet(architecture,
                 data_split,
                 name,
                 pred_sensor,
                 n_sensors,
                 early_stopping,
                 existing_model = False)
        }
        fit_ad_hoc_NN(machine_stack[name], elements, healthy, sensors)
        NeuralNet.evaluation(machine_stack[name])
        plot_loss(machine_stack[name], do_plot_loss)
        save_model(machine_stack[name].model, name)
        NeuralNet.prediction(machine_stack[name], 240)
    H_accuracy = NeuralNet.get_H_score(machine_stack[name], mse_threshold) 

