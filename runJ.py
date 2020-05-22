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
    use = 'AELSTM'
    name = '2'
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
    MLParchitecture = sensors.copy()
    MLParchitecture.update({
        'n_units' : {'first' : 50, 'second' : 15},
        'bias' : True,
        'elements' : [10, 45, 68, 90, 112, 135, 170],
        'healthy' : [33, 43, 52 , 62, 71, 81, 90, 100],
        'epochs' : 50,
        'patience' : 10,
        'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
        'preprocess_type' : 'data',
        'delta' : 4, # Kan ändras
        'n_pattern_steps' : 20, # Kan ändras
        'batch_size' : 25,
        'n_target_steps' : 1,
        'Dense_activation' : 'tanh',
        'pattern_sensors' : ['45','90','135'], # Indices must be used rahter than placements
        'target_sensor' : 2,
        'from' : 0,
        'to' : -1
    })

    LSTMarchitecture = sensors.copy()
    LSTMarchitecture.update({
        'n_units' : {'first' : 50, 'second' : 15},
        'bias' : True,
        'elements' : [10, 45, 68, 90, 112, 135, 170],
        'healthy' : [33, 43, 52 , 62, 71, 81, 90, 100],
        'epochs' : 10,
        'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
        'preprocess_type' : 'data',
        'delta' : 1, # Kan ändras
        'n_pattern_steps' : 150, # Kan ändras
        'batch_size' : 25,
        'n_target_steps' : 20,
        'pattern_delta' : 100,
        'Dense_activation' : 'tanh',
        'LSTM_activation' : 'tanh',
        'pattern_sensors' : ('90'), 
        'target_sensor' : '90',
        'learning_rate' : 0.0001, # 0.001 by default
        'from' : 0,
        'to' : -1
    })

    AELSTMarchitecture = sensors.copy()
    AELSTMarchitecture.update({
        'n_units' : {'first': 800, 'second': 200, 'third' : 40, 'fourth': 20},
        'bias' : True,
        'elements' : [10, 45, 68, 90, 112, 135, 170],
        'healthy' : [33, 43, 52 , 62, 71, 81, 90, 100],
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
        'learning_rate' : 0.01, # 0.001 by default
        'latent_dim' : {'first' : 400, 'second' : 200, 'third' : 40}, 
        'from' : 0,
        'to' : -1
    })
   
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

    healthy_series_stack = {
        '100%' : fit_to_NN(
            architecture['data_split'],
            'our_measurements/healthy/100%/', #2/e90/100%/',
            damaged_element,
            100,
            architecture
        )
    }
    data_split = {'train':0, 'validation':0, 'test':100}

    eval_series_stack = get_eval_series(data_split, damaged_element, architecture, 'our_measurements/e90/') 
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
    

    sensor_to_predict_list = [2] # Påverkar bara titel på plot
    for i in range(len(sensor_to_predict_list)):
        name = 'J_'+use+architecture['name']+str(sensor_to_predict_list[i])
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
            NeuralNet.train_measurements(machine_stack[name], healthy_series_stack['100%'])  
            save_model(machine_stack[name].model, name)
            plot_loss(machine_stack[name], name)
        
        #NeuralNet.evaluation(machine_stack[name], healthy_series_stack['100%'])     
        
        score_stack = {}
        keys = list(eval_series_stack)
        for j in range(len(keys)):
            score_stack.update({
                keys[j] : NeuralNet.evaluation_batch(machine_stack[name], eval_series_stack[keys[j]])
            })    

    plot_performance(score_stack, architecture)
    #binary_prediction = get_binary_prediction(scoreStacks, architecture)
    #plot_roc(prediction)
    ########## PREDICTIONS #############
    
    prediction_manual = {
        'series_to_predict' : 14,
        'stack' : eval_series_stack['52%']
    }
    #prediction = NeuralNet.prediction(machine_stack[name], prediction_manual)
    #plot_prediction(prediction, prediction_manual, use)
    
