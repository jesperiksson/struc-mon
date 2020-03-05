# Modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import h5py

# Classes
from LSTMbatch import LongShortTermMemoryBatch as LSTMbatch

''' Utilities for various classes'''

def fit_H_to_LSTM(data_split, path):
    """
    Function that fits raw data into a format that fits LSTM, i.e. array of arrays of acceleration signals in time-domain
    """
    halfpath = 'measurements/'+ path + 'half/'
    quarterpath = 'measurements/'+ path +'quarter/'
    thirdpath = 'measurements/' + path + 'third/'

    half = os.listdir(halfpath)
    half.sort()
    quarter = os.listdir(quarterpath)
    quarter.sort()
    third = os.listdir(thirdpath)
    third.sort()

    n_files = len(half)
    batchStack = {}

    for i in range(n_files):
        
        halfmat = h5py.File(halfpath + half[i],'r')
        halfdata = halfmat.get('acc')

        quartermat = h5py.File(quarterpath + quarter[i],'r')
        quarterdata = quartermat.get('acc')

        thirdmat = h5py.File(thirdpath + third[i],'r')
        thirddata = thirdmat.get('acc')
        if i/n_files <= data_split[0]/100:
            batchStack.update({
                'batch'+str(i) : LSTMbatch([halfdata[1,:],quarterdata[1,:],thirddata[1,:]],
                                 i,     
                                 category = 'train')
            })
        elif i/n_files > data_split[0]/100 and i/n_files <= (data_split[0]+data_split[1])/100:
            batchStack.update({
                'batch'+str(i) : LSTMbatch([halfdata[1,:],quarterdata[1,:],thirddata[1,:]],
                                 i,
                                 category = 'validation')
            })
        else:
            batchStack.update({
                'batch'+str(i) : LSTMbatch([halfdata[1,:],quarterdata[1,:],thirddata[1,:]],
                                 i,
                                 category = 'test')
            })
    return batchStack

#fit_H_to_LSTM(data_split = [60,20,20])
