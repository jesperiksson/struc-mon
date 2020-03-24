# Modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import model_from_json, Sequential
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Input, RepeatVector, Dropout
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
import h5py

# Classes
from LSTMbatch import LongShortTermMemoryBatch as LSTMbatch
from MLPbatch import *

''' Utilities for various classes'''

def fit_to_NN(data_split, path):
    """
    Function that fits raw data into a format that fits LSTM, i.e. array of arrays of acceleration signals in time-domain
    """
    halfpath = path + 'half/'
    quarterpath = path +'quarter/'
    thirdpath = path + 'third/'

    half = os.listdir(halfpath)
    half.sort()
    quarter = os.listdir(quarterpath)
    quarter.sort()
    third = os.listdir(thirdpath)
    third.sort()

    n_files = len(half)
    batchStack = {}
    scaler = MinMaxScaler(feature_range=(-1,1))
    start = 0
    to = 8000
    diff = to-start
    for i in range(n_files):
        halfmat = h5py.File(halfpath + half[i],'r')
        halfdata = preprocessing.normalize(halfmat.get('acc'))
        #halfdata = scaler.fit_transform(np.array(preprocessing.normalize(halfmat.get('acc')))[:,start:to])

        quartermat = h5py.File(quarterpath + quarter[i],'r')
        quarterdata = preprocessing.normalize(quartermat.get('acc'))
        #quarterdata = scaler.fit_transform(np.array(preprocessing.normalize(quartermat.get('acc')))[:,start:to])
        

        thirdmat = h5py.File(thirdpath + third[i],'r')
        thirddata = preprocessing.normalize(thirdmat.get('acc'))    
        #thirddata = scaler.fit_transform(np.array(preprocessing.normalize(thirdmat.get('acc')))[:,start:to])

        if i/n_files <= data_split[0]/100:
            batchStack.update({
                'batch'+str(i) : LSTMbatch([halfdata[1,:],quarterdata[1,:],thirddata[1,:]],
                                 i,     
                                 diff,
                                 category = 'train')
            })
        elif i/n_files > data_split[0]/100 and i/n_files <= (data_split[0]+data_split[1])/100:
            batchStack.update({
                'batch'+str(i) : LSTMbatch([halfdata[1,:],quarterdata[1,:],thirddata[1,:]],
                                 i,
                                 diff,
                                 category = 'validation')
            })
        else:
            batchStack.update({
                'batch'+str(i) : LSTMbatch([halfdata[1,:],quarterdata[1,:],thirddata[1,:]],
                                 i,
                                 diff,
                                 category = 'test')
            })
    return batchStack

def fit_sin_to_NN(data_split, path):
    batchStack = {}
    datapoints = 200
    n_batches = 1000
    
    x = np.array(range(datapoints))
    for i in range(n_batches):
        amplitude = np.random.rand(1)*0
        if i/n_batches <= data_split[0]/100:
            batchStack.update({
            'batch'+str(i) : MLPbatch(amplitude*[np.array(np.sin(x)), np.array(np.sin(x)), np.array(np.sin(x))],
                             i,
                             category = 'train')
        })
        elif i/n_batches > data_split[0]/100 and i/n_batches <= (data_split[0]+data_split[1])/100:
            batchStack.update({
            'batch'+str(i) : MLPbatch(amplitude*[np.array(np.sin(x)), np.array(np.sin(x)), np.array(np.sin(x))],
                             i,
                             category = 'validation')
        })
        else:
            batchStack.update({
            'batch'+str(i) : MLPbatch(amplitude*[np.array(np.sin(x)), np.array(np.sin(x)), np.array(np.sin(x))],
                             i,
                             category = 'test')
        })
    return batchStack
    
def save_model(model,name):
    '''
    https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    '''
    model_json = model.to_json()
    with open('models/'+name+'.json', 'w') as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    model.save_weights('models/'+name+'.h5')
    print('Saved model:', name)

def plot_prediction(self, prediction, batch_num, hindsight):
    plt.figure()
    plt.plot(range(len(prediction)), prediction, linewidth=0.3)
    plt.plot(range(len(hindsight)), hindsight, linewidth=0.3)
    plt.legend(['Prediction','Data'])
    plt.show()
    return

    

def evaluate(self): # Evaluating the model on the test dataset
    if self.data_split[2] == 0:
        print('No batches assigned for testing')
    '''
    Args:
    '''
    def evaluation_generator():
        test_data = np.array([]);
        i = 0
        while True:
            i += 1
            key = 'batch'+str(i%len(self.batchStack))
            data = self.batchStack[key].data
            if self.batchStack[key].category == 'test':
                test_batch = np.array(self.batchStack[key].data)
                targets = np.reshape(test_batch[self.pred_sensor,:],
                                     [1, np.shape(data)[1], 1])
                patterns = np.reshape(np.delete(test_batch,
                                      self.pred_sensor, axis=0),
                                      [1, np.shape(data)[1], 2])                    
                yield (patterns, targets)
    evaluation = self.model.evaluate_generator(evaluation_generator(), 
                                               steps = int(self.n_batches*self.data_split[2]/100), 
                                               verbose = 1)
    return evaluation

def predict_batch(self, batch_num, sensor):
    pred_batch = self.batchStack['batch'+str(batch_num)].data[sensor]   
    if self.architecture['prediction'] == 'entire_series': 
        patterns = np.reshape(np.delete(pred_batch, self.pred_sensor, axis=0), [1, np.shape(pred_batch)[1], 2])
    elif self.architecture['prediction'] == 'end_of_series':
        patterns = np.array(pred_batch[:-self.architecture['n_pred_units']])  
        patterns = np.reshape(patterns, [1, np.shape(patterns)[0], 1])
        hindsight = np.array(pred_batch[-self.architecture['n_pred_units']:])
        hindsight = np.reshape(hindsight, [np.shape(hindsight)[0], 1]) 
    prediction = self.model.predict_on_batch(patterns)
    prediction = np.reshape(prediction, [self.architecture['n_pred_units'], 1])
    plot_prediction(self,
                    prediction, 
                    batch_num,
                    hindsight) 
    return prediction

def plot_loss(self, show_plot = False):
    plt.figure()
    plt.plot(range(1,self.used_epochs+1), self.loss, 'bo', label='Training loss')
    plt.plot(range(1,self.used_epochs+1), self.val_loss, 'b', label='Validation loss')
    if show_plot == True:
        plt.title('Training and validation loss')
        plt.xlabel = 'Epochs'
        plt.ylabel = self.loss
        plt.legend()
        plt.show()
    else: 
        pass

        

