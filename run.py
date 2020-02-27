# Modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd 

# Class files
from LSTM import * 
from LSTMdata import *

if __name__ == "__main__":


    '''
    Hyperparameters
    '''
    #Network
    n_batches = 5
    train_percentage = 70
    split_mode = 'last' # or random
    pred_sensor = 3 # Which sensor to be the target, the other sensors are patterns
    net_type = 'vanilla'
    #Training
    epochs = 200


#    with open('intervall.csv', 'r') as f:
    data = np.array(pd.read_csv('intervall.csv'))
#        print(data)


    '''Stack of machines'''
    machine_stack = {
        'vanillaLSTM' : LongShortTermMemoryMachine(data, train_percentage, split_mode, pred_sensor,  n_batches , net_type)

    }
    
    vanillaLSTM = machine_stack['vanillaLSTM'].train(epochs)

    prediction = machine_stack['vanillaLSTM'].predict()
    
