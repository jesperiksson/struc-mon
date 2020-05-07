# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tensorflow as tf
import keras
from keras.models import model_from_json, Sequential
#from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Input, RepeatVector, Dropout
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#import pandas as pd
import os
import h5py
import random

# Classes
from MLP import *

''' Utilities for various classes'''

def fit_to_NN_ad_hoc(data_split, path, damaged_element, healthy_percentage):
    from Databatch import DataBatch
    s10path = path + 's10/'
    s45path = path + 's45/'
    s90path = path + 's90/'
    s135path = path + 's135/'
    s170path = path + 's170/'

    seed = 10
    file_list = os.listdir(s10path)
    file_list.sort()
    random.Random(seed).shuffle(file_list)

    speeds = np.empty([len(file_list)])
    for i in range(len(file_list)):
        if len(file_list[i]) == 9:
            speeds[i] = int(file_list[i][0:5])
        elif len(file_list[i]) == 10:
            speeds[i] = int(file_list[i][0:6])
    normalized_speeds = (speeds-min(speeds))/(max(speeds)-min(speeds))

    n_files = int(len(file_list)/4)
    series_stack = {}
    start = 0
    to = -1
    diff = to-start
    for i in range(n_files):
        data = [None]*5
        
        s10mat = h5py.File(s10path + file_list[i],'r')
        data[0] = s10mat.get('acc')

        s45mat = h5py.File(s45path + file_list[i],'r')
        data[1] = s45mat.get('acc')

        s90mat = h5py.File(s90path + file_list[i],'r')
        data[2] = s90mat.get('acc')

        s135mat = h5py.File(s135path + file_list[i],'r')
        data[3] = s135mat.get('acc')

        s170mat = h5py.File(s170path + file_list[i],'r')
        data[4] = s170mat.get('acc')
        
        speed = int(file_list[i][0:5])/1000

        if i/n_files <= data_split[0]/100:
            category = 'train'
        elif i/n_files > data_split[0]/100 and i/n_files <= (data_split[0]+data_split[1])/100:
            category = 'validation'
        else:
            category = 'test'
        series_stack.update({
            'batch'+str(i) : DataBatch([data[0][1,:],
                                        data[1][1,:],
                                        data[2][1,:],
                                        data[3][1,:],
                                        data[4][1,:]],
                             i,
                             speed,
                             normalized_speeds[i],
                             damaged_element,
                             category,
                             healthy_percentage)
                            })
    return series_stack

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

def encode_data(data):
    data = preprocessing.normalize(data.get('acc'))
    return data
# TBD
def decode_data(data):
    return      

def predict_batch(self, batch_num):
    pred_batch = self.series_stack['batch'+str(batch_num)].data[sensor]   
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
    plt.plot(range(1,self.used_epochs+1), self.val_loss, 'ro', label='Validation loss')
    if show_plot == True:
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss - MSE')
        plt.legend()
        plt.show()
    else: 
        pass

def plot_performance3(scoreStacks):
    sensors = ['Half', 'Quarter', 'Third']
    for i in range(len(scoreStacks)):
        scoreStack = scoreStacks[i]
        plt.subplot(len(scoreStacks),1,i+1) 
        plt.plot(scoreStack['H'][1:-1:2],scoreStack['H'][0:-2:2],'b.',label='Healthy data')
        plt.plot(scoreStack['5070'][1:-1:2],scoreStack['5070'][0:-2:2],'r+',label='70% reduction at element 50')
        plt.plot(scoreStack['5090'][1:-1:2],scoreStack['5090'][0:-2:2],'g1',label='90% reduction at element 50')
        plt.plot(scoreStack['7090'][1:-1:2],scoreStack['7090'][0:-2:2],'kv',label='90% reduction at element 70')
        plt.xlabel('Speed [km/h]')
        plt.ylabel('Root Mean Square Error')
        plt.title(sensors[i])
    plt.legend()
    plt.show()

def plot_performance5(scoreStacks):
    sensors = ['1/18', '1/4', '1/2', '3/4', '17/18']
    score_keys = list(scoreStacks)
    cmap = plt.cm.rainbow
    norm = colors.Normalize(vmin=33,vmax=100)
    for i in range(len(score_keys)): # Iterates over sensors
        
        scoreStack = scoreStacks[score_keys[i]]
        plt.subplot(len(scoreStacks),1,i+1)
        percentage_keys = list(scoreStack) 
        for j in range(len(percentage_keys)): # Iterates over percentages
            plt.plot(scoreStack[percentage_keys[j]][1:-1:2], 
                     scoreStack[percentage_keys[j]][0:-2:2], 
                     color=cmap(norm(percentage_keys[j])), 
                     marker='o',
                     linestyle='None')
        plt.xlabel('Speed [km/h]')
        plt.ylabel('Root Mean Square Error')
        plt.title(sensors[score_keys[i]])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm)
    plt.legend()
    plt.show()
        
def plot_prediction(self, prediction, series_num, hindsight):
    plt.figure()
    plt.plot(range(len(prediction)), prediction, linewidth=0.3)
    plt.plot(range(len(hindsight)), hindsight, linewidth=0.3)
    plt.legend(['Prediction','Data'])
    plt.show()
    return
















