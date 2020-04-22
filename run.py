# Modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.io import loadmat
# Class files
''' Imported when needed
from LSTM import * 
from MLP import *
from LSTMbatch import *
'''
# Utility file
from util import *

if __name__ == "__main__":
    # Model
    architecture = {
        'prediction' : 'end_of_series', # end_of_series or entire_series
        'model_type' : 'MLP', # MLP, LSTM or AutoencoderLSTM
        'direction' : 'uni', # uni or bi (uni a.k.a. vanilla net)
        'n_layers' : 1,
        'n_units' : [30, 15],
        'target' : 'accelerations', # E or accelerations
        'bias' : True,
# Metadata
        'elements' : [10, 45, 68, 90, 112, 135, 170],
        'healthy' : [33, 43, 52 , 62, 71, 81, 90, 100],
        'sensors' : [10, 45, 90, 135, 170],
        'speeds' : 20,
        'data_split' : [60, 20, 20], # sorting of data into training testing and validation
# MLP-specific settings
        'delta' : 4,
        'n_pattern_steps' : 10,
        'n_target_steps' : 1,
        'MLPactivation' : 'tanh',
# LSTM-specific settings
        'LSTMactivation' : 'tanh'
    }
    early_stopping = True
    # Data
    elements = [90]#[10, 45, 68, 90, 112, 135, 170]
    healthy = [100]#[33, 43, 52 , 62, 71, 81, 90, 100]
    sensors = [10, 45, 90, 135, 170]
       


    # Training
    epochs = 20

    # Plotting
    do_plot_loss= True  


    if architecture['model_type'] == 'MLP':
        from MLP import *
    elif architecture['model_type'] == 'LSTM':
        from LSTM import *
    elif architecture['model_type'] == 'AELSTM':
        from AELSTM import *

    batchStacks = {
    'H' : fit_to_NN(architecture['data_split'],'measurements/H/'),
    '5070' : fit_to_NN(architecture['data_split'],'measurements/D_50%_70/'),
    '5090' : fit_to_NN(architecture['data_split'],'measurements/D_50%_90/'),
    '7090' : fit_to_NN(architecture['data_split'],'measurements/D_70%_90/'),
    }
    machine_stack = {}
    scoreStacks = {}
    sensor_to_predict = [0]
    for i in range(len(sensor_to_predict)):
        name = 'MLPmodel3_3_bias_10_'+str(sensor_to_predict[i])
        try:
            f = open('models/'+name+'.json')
            machine_stack.update({
                name : NeuralNet(architecture,
                     name,
                     early_stopping,
                     existing_model = True,
                     sensor_to_predict = i)
            })
        except IOError:    
            machine_stack.update({
                name : NeuralNet(architecture,
                     name,
                     early_stopping,
                     existing_model = False,
                     sensor_to_predict = i)
            })
            NeuralNet.train_measurements(machine_stack[name], batchStacks['H'], epochs)
            NeuralNet.evaluation(machine_stack[name], batchStacks['H'])
            plot_loss(machine_stack[name], do_plot_loss)
            save_model(machine_stack[name].model, name)
        
        scoreStacks.update({sensor_to_predict[i] : {
        'H' : NeuralNet.evaluation_batch(machine_stack[name], batchStacks['H']),
        '5070' : NeuralNet.evaluation_batch(machine_stack[name], batchStacks['5070']),
        '5090' : NeuralNet.evaluation_batch(machine_stack[name], batchStacks['5090']),
        '7090' : NeuralNet.evaluation_batch(machine_stack[name], batchStacks['7090'])
        }})
    #plot_performance(scoreStacks)
    prediction, hindsight = NeuralNet.prediction(machine_stack[name], batchStacks['7090'], 200)
    plot_prediction(machine_stack[name],prediction, 200, hindsight)

