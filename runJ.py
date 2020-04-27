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
        'sensors' : [90],#[45, 90, 135]
        'speeds' : 20,
        'data_split' : [60, 20, 20], # sorting of data into training testing and validation
# MLP-specific settings
        'delta' : 4, # Kan ändras
        'n_pattern_steps' : 10, # Kan ändras
        'n_target_steps' : 1,
        'MLPactivation' : 'tanh',
# LSTM-specific settings
        'LSTMactivation' : 'tanh'
    }
    early_stopping = True
    # Data
    damaged_element = 90#[10, 45, 68, 90, 112, 135, 170] Finns data på 45, 90 och 135
       
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

    healthy_batch_stack = {
    '100%' : fit_to_NN_ad_hoc(architecture['data_split'],
                              'our_measurements/e90/100%/',
                              damaged_element,
                              100)}
    eval_batch_stack = {
    '90%' : fit_to_NN_ad_hoc([0,0,100],
                            'our_measurements/e'+str(damaged_element)+'/90%/',
                            damaged_element,
                            90),
    '81%' : fit_to_NN_ad_hoc([0,0,100],
                            'our_measurements/e'+str(damaged_element)+'/81%/',
                            damaged_element,
                            81),
    '71%' : fit_to_NN_ad_hoc([0,0,100],
                            'our_measurements/e'+str(damaged_element)+'/71%/',
                            damaged_element,
                            71),
    '62%' : fit_to_NN_ad_hoc([0,0,100],
                        'our_measurements/e'+str(damaged_element)+'/62%/',
                        damaged_element,
                        62),     
    '52%' : fit_to_NN_ad_hoc([0,0,100],
                        'our_measurements/e'+str(damaged_element)+'/52%/',
                        damaged_element,
                        52),
    '43%' : fit_to_NN_ad_hoc([0,0,100],
                        'our_measurements/e'+str(damaged_element)+'/43%/',
                        damaged_element,
                        43),
    '33%' : fit_to_NN_ad_hoc([0,0,100],
                        'our_measurements/e'+str(damaged_element)+'/33%/',
                        damaged_element,
                        33)
                        }
    machine_stack = {}
    scoreStacks = {}
    sensor_to_predict_list = [2] # Påverkar bara titel på plot
    for i in range(len(sensor_to_predict_list)):
        name = 'J_MLPmodel3_a'+str(sensor_to_predict_list[i])
        try:
            f = open('models/'+name+'.json')
            machine_stack.update({
                name : NeuralNet(architecture,
                     name,
                     early_stopping,
                     existing_model = True,
                     sensor_to_predict = sensor_to_predict_list[i])
            })
        except IOError:    
            machine_stack.update({
                name : NeuralNet(architecture,
                     name,
                     early_stopping,
                     existing_model = False,
                     sensor_to_predict = sensor_to_predict_list[i])
            })
            NeuralNet.train_measurements(machine_stack[name], healthy_batch_stack['100%'], epochs)
            NeuralNet.evaluation(machine_stack[name], healthy_batch_stack['100%'])
            plot_loss(machine_stack[name], do_plot_loss)
            save_model(machine_stack[name].model, name)        
        scoreStacks.update({sensor_to_predict_list[i] : {
        100: NeuralNet.evaluation_batch(machine_stack[name], healthy_batch_stack['100%']),
        90 : NeuralNet.evaluation_batch(machine_stack[name], eval_batch_stack['90%']),
        81 : NeuralNet.evaluation_batch(machine_stack[name], eval_batch_stack['81%']),
        71 : NeuralNet.evaluation_batch(machine_stack[name], eval_batch_stack['71%']),
        62 : NeuralNet.evaluation_batch(machine_stack[name], eval_batch_stack['62%']),
        52 : NeuralNet.evaluation_batch(machine_stack[name], eval_batch_stack['52%']),
        43 : NeuralNet.evaluation_batch(machine_stack[name], eval_batch_stack['43%']),
        33 : NeuralNet.evaluation_batch(machine_stack[name], eval_batch_stack['33%'])
        }})
        
    plot_performance5(scoreStacks)
    #prediction, hindsight = NeuralNet.prediction(machine_stack[name], healthy_batch_stack['100%'], 200)
    #plot_prediction(machine_stack[name],prediction, 200, hindsight)
