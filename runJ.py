# Modules
#import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#import matplotlib.pyplot as plt
#from scipy.io import loadmat
# Class files
from Databatch import *
# Utility file
from util import *

if __name__ == "__main__":
    # Which model to use (MLP or LSTM):
    #####################
    use = 'LSTM'
    name = 'peaks_two_layers_set3'
    #####################

    sensors = {
        'name' : name,        
        'active_sensors' : ['90']
        }
    sensor_dict = {}
    for i in range(len(sensors['active_sensors'])):
        sensor_dict.update({
            sensors['active_sensors'][i] : i
            })
    sensors.update({
        'sensors' : sensor_dict
        })
    if use == 'MLP':
        architecture = sensors
        from MLP import *
        architecture.update({
            # Net configuration
            'bias' : True,
            'n_pattern_steps' : 20, # Kan ändras
            'n_target_steps' : 1,
            'n_units' : {'first' : 50, 'second' : 15},
            # Sensor parameters
            'pattern_sensors' : ['90'], # Indices must be used rahter than placements
            'target_sensor' : '90',
            'target_sensors' : ['90'],
            # Training parameters
            'Dense_activation' : 'tanh',
            'epochs' : 50,
            'patience' : 10,
            'early_stopping' : True,
            'learning_rate' : 0.001, # 0.001 by default
            'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
            'preprocess_type' : 'peaks',
            'delta' : 2,      
            'batch_size' : 25,
            # Data interval
            'from' : 0,
            'to' : -1
        })
    elif use == 'LSTM':
        architecture = sensors
        from LSTM import *
        architecture.update({
            'model' : '7',
            # Net configuaration
            'n_units' : {'first' : 150, 'second' : 150},
            'bias' : True,
            'n_pattern_steps' : 200, # Kan ändras
            'n_target_steps' : 50,
            # Sensor parameters
            'pattern_sensors' : ['90'],
            'target_sensor' : '90',
            'target_sensors' : ['90'],
            # Training parameters
            'batch_size' : 1,
            'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
            'delta' : 1, # Kan ändras
            'Dense_activation' : 'tanh',
            'early_stopping' : True,
            'epochs' : 200,
            'learning_rate' : 0.001, # 0.001 by default
            'LSTM_activation' : 'tanh',
            'preprocess_type' : 'peaks',
            'patience' : 30,
            'pattern_delta' : 10,
            # Data interval
            'from' : 2000,
            'to' : 15000,
            # Model saving
            'save_periodically' : True,
            'save_interval' : 10 # Number of series to train on before saving
        })
    elif use == 'AELSTM':
        architecture = sensor
        from AELSTM import *
        architecture.update({
            'n_units' : {'first': 800, 'second': 200, 'third' : 40, 'fourth': 20},
            'bias' : True,
            'speeds' : 20,
            'epochs' : 10,
            'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
            'preprocess_type' : 'data',
            'delta' : 1, # Kan ändras
            'n_pattern_steps' : 400, # Kan ändras
            'batch_size' : 16,
            'n_target_steps' : 400,
            'pattern_delta' : 50,
            'Dense_activation' : 'tanh',
            'LSTM_activation' : 'tanh',
            'learning_rate_schedule' : False,
            'pattern_sensors' : ['90'], 
            'target_sensor' : '90',
            'target_sensors' : ['90'],
            'learning_rate' : 0.01, # 0.001 by default
            'early_stopping' : True,
            'latent_dim' : {'first' : 400, 'second' : 200, 'third' : 40}, 
            'from' : 0,
            'to' : -1,
            # Model saving
            'save_periodically' : True,
            'save_interval' : 10 # Number of series to train on before saving
        })
   

    healthy_series_stack = {
        '100%' : fit_to_NN(
            architecture['data_split'],
            'our_measurements3/e90/100%/',#2/e90/100%/'
            100,
            architecture
        )
    }
    data_split = {'train':0, 'validation':0, 'test':100}

    eval_series_stack = get_eval_series(data_split, architecture, 'our_measurements/e90/') 
    #DataBatch.plot_series(healthy_series_stack['100%']['batch36'], plot_sensor = ['1/2'])
    #DataBatch.plot_frequency(eval_series_stack['52%']['frequency'], sensors)
    #DataBatch.plot_batch(healthy_series_stack['100%']['data'], architecture)
    #DataBatch.plot_batch(eval_series_stack['90%'])
    #DataBatch.plot_batch(eval_series_stack['81%']['data'], architecture)
    #DataBatch.plot_batch(eval_series_stack['71%'])
    #DataBatch.plot_batch(eval_series_stack['62%'])
    #DataBatch.plot_batch(eval_series_stack['52%'])
    #DataBatch.plot_batch(eval_series_stack['43%'])
    #DataBatch.plot_batch(eval_series_stack['33%'])
    #quit()
    
    machine_stack = {}
    
    for i in range(len(architecture['target_sensors'])):
        architecture['target_sensor'] = architecture['target_sensors'][i]
        name = 'J_'+use+architecture['name']+architecture['target_sensor']
        try:
            f = open('models/'+name+'.json')
            machine_stack.update({
                name : NeuralNet(architecture,
                     name,
                     existing_model = True)
            })
        except IOError:    
            machine_stack.update({
                name : NeuralNet(architecture,
                     name,
                     existing_model = False)
            })
            NeuralNet.train(machine_stack[name], healthy_series_stack['100%'])  
            save_model(machine_stack[name].model, name)
            plot_loss(machine_stack[name], name)
        
        #NeuralNet.evaluation(machine_stack[name], healthy_series_stack['100%'])     
        
        score_stack = {}
        keys = list(eval_series_stack)
        '''
        for j in range(len(keys)):
            score_stack.update({
                keys[j] : NeuralNet.evaluation_batch(machine_stack[name], eval_series_stack[keys[j]])
            })    
        '''
    #plot_performance(score_stack, architecture)
    #binary_prediction = get_binary_prediction(scoreStacks, architecture)
    #plot_roc(prediction)
    ########## PREDICTIONS #############
    
    prediction_manual = {
        'series_to_predict' : 5,
        'stack' : eval_series_stack['52%']
    }
    #prediction = NeuralNet.prediction(machine_stack[name], prediction_manual)
    #plot_prediction(prediction, prediction_manual, use)
    forecast = NeuralNet.forecast(machine_stack, prediction_manual)
    plot_forecast(forecast, prediction_manual, architecture)

    
