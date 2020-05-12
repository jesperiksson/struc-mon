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
    # Which model to use (MLP or LSTM):
    #####################
    use = 'MLP'
    #####################

    MLParchitecture = {
        'n_units' : [30, 15],
        'bias' : True,
        'elements' : [10, 45, 68, 90, 112, 135, 170],
        'healthy' : [33, 43, 52 , 62, 71, 81, 90, 100],
        'sensors' : [10, 45, 90, 135, 170],
        'speeds' : 20,
        'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
        'delta' : 1, # Kan ändras
        'n_pattern_steps' : 5, # Kan ändras
        'batch_size' : 25,
        'n_target_steps' : 1,
        'Dense_activation' : 'tanh',
        'pattern_sensors' : [2], # Indices must be used rahter than placements
        'target_sensor' : 2
    }

    LSTMarchitecture = {
        'n_units' : [100, 65, 35],
        'bias' : True,
        'elements' : [10, 45, 68, 90, 112, 135, 170],
        'healthy' : [33, 43, 52 , 62, 71, 81, 90, 100],
        'sensors' : [10, 45, 90, 135, 170],
        'speeds' : 20,
        'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
        'delta' : 1, # Kan ändras
        'n_pattern_steps' : 250, # Kan ändras
        'batch_size' : 25,
        'n_target_steps' : 50,
        'pattern_delta' : 100,
        'Dense_activation' : 'tanh',
        'LSTM_activation' : 'tanh',
        'pattern_sensors' : [2], # Indices must be used rahter than placements
        'target_sensor' : 2,
        'learning_rate' : 0.0001 # 0.001 by default
    }

    AELSTMarchitecture = {
        'n_units' : [100, 65, 35],
        'bias' : True,
        'elements' : [10, 45, 68, 90, 112, 135, 170],
        'healthy' : [33, 43, 52 , 62, 71, 81, 90, 100],
        'sensors' : [10, 45, 90, 135, 170],
        'speeds' : 20,
        'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
        'delta' : 1, # Kan ändras
        'n_pattern_steps' : 2000, # Kan ändras
        'batch_size' : 25,
        'n_target_steps' : 1000,
        'pattern_delta' : 2000,
        'Dense_activation' : 'tanh',
        'LSTM_activation' : 'tanh',
        'pattern_sensors' : [2], # Indices must be used rahter than placements
        'target_sensor' : 2,
        'learning_rate' : 0.0001, # 0.001 by default
        'latent_dim' : [100,50] 
    }
   
    if use == 'MLP':
        architecture = MLParchitecture
        from MLP import *
    elif use == 'LSTM':
        architecture = LSTMarchitecture
        from LSTM import *
    elif use == 'AELSTM':
        architecture = AELSTMarchitecture
        from AELSTM import *

    early_stopping = False
    # Data
    damaged_element = 90#[10, 45, 68, 90, 112, 135, 170] Finns data på 45, 90 och 135
       
    # Training
    epochs = 50

    healthy_series_stack = {
    '100%' : fit_to_NN_ad_hoc(architecture['data_split'],
                              'our_measurements/healthy/100%/',
                              damaged_element,
                              100)}
    data_split = {'train':0, 'validation':0, 'test':100}
    '''

    '''
    eval_series_stack = get_eval_series(data_split, damaged_element) 
    #DataBatch.plot_series(healthy_series_stack['100%']['batch36'], plot_sensor = ['1/2'])
    #DataBatch.plot_batch(healthy_series_stack['100%'])
    DataBatch.plot_batch(eval_series_stack['90%'])
    DataBatch.plot_batch(eval_series_stack['81%'])
    DataBatch.plot_batch(eval_series_stack['71%'])
    DataBatch.plot_batch(eval_series_stack['62%'])
    DataBatch.plot_batch(eval_series_stack['52%'])
    DataBatch.plot_batch(eval_series_stack['43%'])
    DataBatch.plot_batch(eval_series_stack['33%'])
    quit()
    machine_stack = {}
    scoreStacks = {}

    sensor_to_predict_list = [2] # Påverkar bara titel på plot
    for i in range(len(sensor_to_predict_list)):
        name = 'J_'+use+'model5_2_'+str(sensor_to_predict_list[i])
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
            #save_model(machine_stack[name].model, name+'mod') 
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
            plot_loss(machine_stack[name], name)
        
        NeuralNet.evaluation(machine_stack[name], healthy_series_stack['100%'])
         
        eval_series_stack = get_eval_series(data_split, damaged_element)               
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
        
    plot_performance5(scoreStacks)

    ########## PREDICTIONS #############
    prediction_manual = {
        'series_to_predict' : 1,
        'stack' : healthy_series_stack['100%']
    }
    prediction = NeuralNet.prediction(machine_stack[name], prediction_manual)
    plot_prediction(prediction, prediction_manual)
