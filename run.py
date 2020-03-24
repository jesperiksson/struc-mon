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
        'sensor_key' : 'half', # half, quarter or third. For end_of_series it specifies which series'end to predict, for entire_series it specifies which entire serie
        'model_type' : 'MLP', # MLP, LSTM or AutoencoderLSTM
        'direction' : 'uni', # uni or bi (uni a.k.a. vanilla net)
        'n_layers' : 1,
        'n_units' : [8, 3],
        'sensor_list' : ['half', 'quarter', 'third'],
        'n_pred_units' : 50, # Number of units to be predicted
# MLP-specific settings
        'delta' : 10,
        'n_pattern_steps' : 20,
        'n_target_steps' : 1,
        'MLPactivation' : 'relu'
# LSTM-specific settings
    }
    feature_wise_normalization = False
    early_stopping = True
    # Data
    train_percentage = 60
    test_percentage = 20
    validation_percentage = 20
    data_split = [train_percentage, test_percentage, validation_percentage]
    assert sum(data_split) == 100
    split_mode = 'last' # or shuffle
    
    sensor_dict = {
        'half' : 0,
        'quarter' : 1,
        'third' : 2}
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

    H_batchStack = fit_to_NN(data_split, path = 'measurements/H/')
    D_50_70_batchStack = fit_to_NN([0, 100, 0], path = 'measurements/D_50%_70/')
    D_50_90_batchStack = fit_to_NN([0, 100, 0], path = 'measurements/D_50%_90/')
    D_70_90_batchStack = fit_to_NN([0, 100, 0], path = 'measurements/D_70%_90/')
    #batchStack['batch1'].plot_batch(architecture['sensor_list'], 1)
    #batchStack = fit_sin_to_NN(data_split, path = 'measurements_sin_csv/')
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
                 H_batchStack,
                 data_split,
                 name,
                 pred_sensor,
                 n_sensors,
                 feature_wise_normalization,
                 early_stopping,
                 existing_model = True)
        }
    except IOError:    
        machine_stack = {
            name : NeuralNet(architecture,
                 H_batchStack,
                 data_split,
                 name,
                 pred_sensor,
                 n_sensors,
                 feature_wise_normalization,
                 early_stopping,
                 existing_model = False)
        }
        NeuralNet.train(machine_stack[name], epochs) # For Time-series MLP change sizes locally        
        NeuralNet.evaluation(machine_stack[name])
        plot_loss(machine_stack[name], do_plot_loss)
        save_model(machine_stack[name].model, name)
        NeuralNet.prediction(machine_stack[name], 240)
    H_accuracy = NeuralNet.get_H_score(machine_stack[name], mse_threshold) 
    print('H_accuracy score: ', H_accuracy)
    D_50_70_score = NeuralNet.get_D_score(machine_stack[name], mse_threshold, D_50_70_batchStack)
    D_50_90_score = NeuralNet.get_D_score(machine_stack[name], mse_threshold, D_50_90_batchStack)
    D_70_90_score = NeuralNet.get_D_score(machine_stack[name], mse_threshold, D_70_90_batchStack)
    print(D_50_70_score, D_50_90_score, D_70_90_score)
    
