# Modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.io import loadmat
#from datetime import datetime, date, time

# Class files
from LSTM import * 
from LSTMbatch import *

# Utility file
from util import *

if __name__ == "__main__":


    '''
    Hyperparameters
    '''
    #Model
    architecture = {
        'direction' : 'uni', # uni or bi (uni a.k.a. vanilla net)
        'n_LSTM_layers' : 2,
        'n_units' : [200, 200, 100]
    }
    train_percentage = 60
    test_percentage = 20
    validation_percentage = 20
    data_split = [train_percentage, test_percentage, validation_percentage]
    assert sum(data_split) == 100
    split_mode = 'last' # or shuffle
    sensor_key = 'third'
    sensor_dict = {
        'half' : 0,
        'quarter' : 1,
        'third' : 2}
    pred_sensor = sensor_dict[sensor_key] # Which sensor to be the target, the other sensors are patterns
    n_sensors = len(sensor_dict)
    feature_wise_normalization = True # TODO
    early_stopping = True
    #Training
    epochs = 2

    name = str(train_percentage)+str(test_percentage)+str(validation_percentage)+split_mode+sensor_key+str(epochs)

    batchStack = fit_H_to_LSTM(data_split, path = 'H/')

    '''Stack of machines'''
    machine_stack = {
        'HLSTM' : LongShortTermMemoryMachine(architecture,
                                             batchStack,
                                             data_split,
                                             pred_sensor,
                                             n_sensors,
                                             feature_wise_normalization,
                                             early_stopping
                                             )
    }
    
    HLSTM = machine_stack['HLSTM'].train(epochs)
    machine_stack['HLSTM'].plot_loss()
    evaluation = machine_stack['HLSTM'].evaluate()
    save_model(machine_stack['HLSTM'].model, name)
    
