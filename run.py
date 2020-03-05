# Modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.io import loadmat
from datetime import datetime, date, time

# Class files
from LSTM import * 
from LSTMbatch import *

# Utility file
from util import *

if __name__ == "__main__":


    '''
    Hyperparameters
    '''
    #Network
    train_percentage = 60
    test_percentage = 20
    validation_percentage = 20
    n_batches = 20
    data_split = [train_percentage, test_percentage, validation_percentage]
    assert sum(data_split) == 100
    split_mode = 'last' # or random
    pred_sensor = 2 # Which sensor to be the target, the other sensors are patterns
    net_type = 'vanilla'
    feature_wise_normalization = True
    #Training
    epochs = 200

    batchStack = fit_H_to_LSTM(data_split, path = 'H/')

    '''Stack of machines'''
    machine_stack = {
        'HLSTM' : LongShortTermMemoryMachine(batchStack, pred_sensor, net_type, feature_wise_normalization)
        

    }
    
    HLSTM = machine_stack['HLSTM'].train(epochs)

    prediction = machine_stack['HLSTM'].predict()
    
