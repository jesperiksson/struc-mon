# Modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from util import *
import pandas as pd 

# Class files
from LSTM import * 

if __name__ == "__main__":

    with open('intervall.csv', 'r') as f:
        data = np.array(pd.read_csv('intervall.csv'))
        for row in data:
            print(row)
    trainset = data[:,:150]
    testset = data[:,150:]

    '''Stack of machines'''
    machine_stack = {
        'vanillaLSTM' : LongShortTermMemoryMachine(trainset, testset, n_batches = 5 , net_type = 'vanilla')

    }

    print("\nStarting a LSTM machine")
    LSTM = LongShortTermMemoryMachine(LSTMdata, n_samples)
    LSTM.trainLSTM(LSTMdata) # To be cgit ontinued
