# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

    seed = 1
    s10 = os.listdir(s10path)
    s10.sort()
    random.Random(seed).shuffle(s10)

    s45 = os.listdir(s45path)
    s45.sort()
    random.Random(seed).shuffle(s45)

    s90 = os.listdir(s90path)
    s90.sort()
    random.Random(seed).shuffle(s90)

    s135 = os.listdir(s135path)
    s135.sort()
    random.Random(seed).shuffle(s135)

    s170 = os.listdir(s170path)
    s170.sort()
    random.Random(seed).shuffle(s170)

    file_list = s90
    speeds = np.empty([len(file_list)])
    for i in range(len(file_list)):
        if len(file_list[i]) == 9:
            speeds[i] = int(file_list[i][0:5])
        elif len(file_list[i]) == 10:
            speeds[i] = int(file_list[i][0:6])
    normalized_speeds = (speeds-min(speeds))/(max(speeds)-min(speeds))

    n_files = len(s90)
    batchStack = {}
    start = 0
    to = 10000
    diff = to-start
    for i in range(n_files):
        data = [None]*5
        
        s10mat = h5py.File(s10path + s10[i],'r')
        data[0] = encode_data(s10mat)

        s45mat = h5py.File(s45path + s45[i],'r')
        data[1] = encode_data(s45mat)

        s90mat = h5py.File(s90path + s90[i],'r')
        data[2] = encode_data(s90mat)

        s135mat = h5py.File(s135path + s135[i],'r')
        data[3] = encode_data(s135mat)

        s170mat = h5py.File(s170path + s170[i],'r')
        data[4] = encode_data(s170mat)
        
        speed = int(file_list[i][0:5])/1000

        if i/n_files <= data_split[0]/100:
            category = 'train'
        elif i/n_files > data_split[0]/100 and i/n_files <= (data_split[0]+data_split[1])/100:
            category = 'validation'
        else:
            category = 'test'
        batchStack.update({
            'batch'+str(i) : DataBatch([data[0][1,start:to],
                                        data[1][1,start:to],
                                        data[2][1,start:to],
                                        data[3][1,start:to],
                                        data[4][1,start:to]],
                             i,
                             speed,
                             normalized_speeds[i],
                             damaged_element,
                             category,
                             healthy_percentage)
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

def encode_data(data):
    data = preprocessing.normalize(data.get('acc'))
    return data
# TBD
def decode_data(data):
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

def predict_batch(self, batch_num):
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
        


















