# Modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.io import loadmat
# Class files
from Databatch import DataBatch
# Utility file
from util import *

if __name__ == "__main__":
    # Model
    architecture = {
        'prediction' : 'end_of_series', # end_of_series or entire_series
        'model_type' : 'LSTM', # MLP, LSTM or AutoencoderLSTM
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
        'delta' : 1, # Kan ändras
        'n_pattern_steps' : 200, # Kan ändras
        'batch_size' : 25,
        'n_target_steps' : 100,
        'pattern_delta' : 100,
        'Dense_activation' : 'tanh',
        'LSTM_activation' : 'tanh',
# LSTM-specific settings
        'LSTMactivation' : 'tanh',
        'pattern_sensors' : [2], # Indices must be used rahter than placements
        'target_sensor' : 2
    }
    #architecture.update('n_series' : )
    early_stopping = True
    # Data
    damaged_element = 90#[10, 45, 68, 90, 112, 135, 170] Finns data på 45, 90 och 135
       
    # Training
    epochs = 1

    # Plotting
    do_plot_loss= True  


    if architecture['model_type'] == 'MLP':
        from MLP import *
    elif architecture['model_type'] == 'LSTM':
        from LSTM import *
    elif architecture['model_type'] == 'AELSTM':
        from AELSTM import *

    healthy_series_stack = {
    '100%' : fit_to_NN_ad_hoc(architecture['data_split'],
                              'our_measurements/healthy/100%/',
                              damaged_element,
                              100)}
    
    eval_series_stack = {
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
    
    #DataBatch.plot_series(healthy_series_stack['100%']['batch1'], plot_sensor = ['1/2'])
    #DataBatch.plot_batch(healthy_series_stack['100%']['batch1'], sensor_list = ['1/2'])
    
    machine_stack = {}
    scoreStacks = {}
    sensor_to_predict_list = [2] # Påverkar bara titel på plot
    for i in range(len(sensor_to_predict_list)):
        name = 'J_'+architecture['model_type']+'model4'+str(sensor_to_predict_list[i])
        try:
            f = open('models/'+name+'.json')
            machine_stack.update({
                name : NeuralNet(architecture,
                     name,
                     early_stopping,
                     existing_model = True,
                     sensor_to_predict = sensor_to_predict_list[i])
            })
            #NeuralNet.modify_model(machine_stack[name].model) # OBS!!!
            save_model(machine_stack[name].model, name+'mod') 
        except IOError:    
            machine_stack.update({
                name : NeuralNet(architecture,
                     name,
                     early_stopping,
                     existing_model = False,
                     sensor_to_predict = sensor_to_predict_list[i])
            })
            NeuralNet.train_measurements(machine_stack[name], healthy_series_stack['100%'], epochs)
            save_model(machine_stack[name].model, name)
        ''' 
        NeuralNet.evaluation(machine_stack[name], healthy_series_stack['100%'])
        #plot_loss(machine_stack[name], do_plot_loss)                  
        scoreStacks.update({sensor_to_predict_list[i] : {
        100: NeuralNet.evaluation_batch(machine_stack[name], healthy_series_stack['100%']),
        90 : NeuralNet.evaluation_batch(machine_stack[name], eval_series_stack['90%']),
        81 : NeuralNet.evaluation_batch(machine_stack[name], eval_series_stack['81%']),
        71 : NeuralNet.evaluation_batch(machine_stack[name], eval_series_stack['71%']),
        62 : NeuralNet.evaluation_batch(machine_stack[name], eval_series_stack['62%']),
        52 : NeuralNet.evaluation_batch(machine_stack[name], eval_series_stack['52%']),
        43 : NeuralNet.evaluation_batch(machine_stack[name], eval_series_stack['43%']),
        33 : NeuralNet.evaluation_batch(machine_stack[name], eval_series_stack['33%'])
        }})
        '''
    #plot_performance5(scoreStacks)
    #print(machine_stack[name], healthy_series_stack['100%'])
    ########## PREDICTIONS #############
    series_to_predict = 1
    prediction = NeuralNet.prediction(machine_stack[name], healthy_series_stack['100%'], series_to_predict)
    plot_prediction(machine_stack[name],prediction, series_to_predict, hindsight)
