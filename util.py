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
import random

# Classes
from MLP import *

''' Utilities for various classes'''

def fit_to_NN(data_split, path):
    from Databatch_measurements import DataBatch
    halfpath = path + 'half/'
    quarterpath = path +'quarter/'
    thirdpath = path + 'third/'

    seed = 1337
    half = os.listdir(halfpath)
    half.sort()
    random.Random(seed).shuffle(half)

    quarter = os.listdir(quarterpath)
    quarter.sort()
    random.Random(seed).shuffle(quarter)

    third = os.listdir(thirdpath)
    third.sort()
    random.Random(seed).shuffle(third)

    file_list = half
    speeds = np.empty([len(file_list)])
    for i in range(len(file_list)):
        speeds[i] = int(file_list[i][-18:-12])
    normalized_speeds = (speeds-min(speeds))/(max(speeds)-min(speeds))      

    n_files = len(half)
    batchStack = {}
    start = 2000
    to = 6000
    diff = to-start
    for i in range(n_files):
        halfmat = h5py.File(halfpath + half[i],'r')
        halfdata = encode_data(halfdata)

        quartermat = h5py.File(quarterpath + quarter[i],'r')
        quarterdata = encode_data(quarterdata)       

        thirdmat = h5py.File(thirdpath + third[i],'r')
        thirddata = encode_data(thirddata) 

        speed = int(file_list[i][-18:-12])/1000
        if i/n_files <= data_split[0]/100:
            batchStack.update({
                'batch'+str(i) : DataBatch([halfdata[1,start:to],quarterdata[1,start:to],thirddata[1,start:to]],
                                 i,     
                                 speed,
                                 normalized_speeds[i],
                                 category = 'train')
            })
        elif i/n_files > data_split[0]/100 and i/n_files <= (data_split[0]+data_split[1])/100:
            batchStack.update({
                'batch'+str(i) : DataBatch([halfdata[1,start:to],quarterdata[1,start:to],thirddata[1,start:to]],
                                 i,
                                 speed,
                                 normalized_speeds[i],
                                 category = 'validation')
            })
        else:
            batchStack.update({
                'batch'+str(i) : DataBatch([halfdata[1,start:to],quarterdata[1,start:to],thirddata[1,start:to]],
                                 i,
                                 speed,
                                 normalized_speeds[i],
                                 category = 'test')
            })
    return batchStack


def fit_ad_hoc_NN(machine, elements, healthy, sensors, path = None):
    data = np.empty([])
    path1 = './newly_generated_measurements'
    batchStack = {}
    if path == None:
        for i in range(len(elements)):
            path2 = '/e'+str(elements[i])
            for j in range(len(healthy)):
                path3 = '/'+str(healthy[j])+'%'
                path = path1+path2+path3+'/'
                speeds = os.listdir(path +'s10')
                speeds.sort()
                element = elements[i]
                s10path = path+'s10/'
                s45path = path+'s45/'
                s90path = path+'s90/'
                s135path = path+'s135/'
                s170path = path+'s170/'
                for l in range(len(speeds)):
                    speed = 0
                    if i%int(machine.architecture['data_split'][1]/100*len(speeds)) == 0:
                        category = 'validation'
                    else:
                        category = 'train'
                    s10 = preprocessing.normalize(h5py.File(s10path+speeds[l],'r').get('acc'))
                    s45 = preprocessing.normalize(h5py.File(s45path+speeds[l],'r').get('acc'))
                    s90 = preprocessing.normalize(h5py.File(s90path+speeds[l],'r').get('acc'))
                    s135 = preprocessing.normalize(h5py.File(s135path+speeds[l],'r').get('acc'))
                    s170 = preprocessing.normalize(h5py.File(s170path+speeds[l],'r').get('acc'))
                    batchStack.update({'batch'+str(i) : DataBatch([ np.array(s10[1,:]),
                                                                    np.array(s45[1,:]),
                                                                    np.array(s90[1,:]),
                                                                    np.array(s135[1,:]),
                                                                    np.array(s170[1,:])],
                                        l,
                                        speed,
                                        element,
                                        category,
                                        damage_state = healthy[j])
                    })
                        
                
                NeuralNet.train_ad_hoc(machine, batchStack, epochs=200)
    else: # when a path is provided 
        speed = os.listdir(path +'s10')
        speed.sort()
        s10path = path+'s10/'
        s45path = path+'s45/'
        s90path = path+'s90/'
        s135path = path+'s135/'
        s170path = path+'s170/'
        for i in range(len(speed)):
            if i%int(machine.architecture['data_split'][1]/100*len(speed)) == 0:
                category = 'validation'
            else:
                category = 'train'
            s10mat = preprocessing.normalize(h5py.File(s10path+speed[i],'r').get('acc'))
            s45mat = preprocessing.normalize(h5py.File(s45path+speed[i],'r').get('acc'))
            s90mat = preprocessing.normalize(h5py.File(s90path+speed[i],'r').get('acc'))
            s135mat = preprocessing.normalize(h5py.File(s135path+speed[i],'r').get('acc'))
            s170mat = preprocessing.normalize(h5py.File(s170path+speed[i],'r').get('acc'))
            batchStack.update({
            'batch'+str(i) :    Databatch([s10[1,:], s45[1,:], s90[1,:], s135[1,:], s170[1,:]],
                                l,
                                speed,
                                element,
                                category,
                                damage_state = healthy[j])
            })
                
        
        NeuralNet.train_ad_hoc(machine, batchStack, epochs=200)

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

def plot_performance(scoreStacks):
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
        


















