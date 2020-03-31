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
from LSTMbatch import DataBatch 
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
                'batch'+str(i) : DataBatch([halfdata[1,:],quarterdata[1,:],thirddata[1,:]],
                                 i,     
                                 diff,
                                 speed,
                                 category = 'train')
            })
        elif i/n_files > data_split[0]/100 and i/n_files <= (data_split[0]+data_split[1])/100:
            batchStack.update({
                'batch'+str(i) : DataBatch([halfdata[1,:],quarterdata[1,:],thirddata[1,:]],
                                 i,
                                 speed,
                                 diff,
                                 category = 'validation')
            })
        else:
            batchStack.update({
                'batch'+str(i) : DataBatch([halfdata[1,:],quarterdata[1,:],thirddata[1,:]],
                                 i,
                                 speed,
                                 diff,
                                 category = 'test')
            })
    return batchStack


def fit_ad_hoc_NN(machine, elements, healthy, sensors, speeds, path = None):
    data = np.empty([])
    path1 = '/newly_generated_measurements'
    batchStack = {}
    if path == None:
        for i in range(len(elements)):
            path2 = 'e'+str(elements(i))
            for j in range(len(healthy)):
                path3 = str(healthy(j))+'%'
                path = path1+path2+path3
                speed = sort(os.listdir(path +'s10'))
                s10path = path+'s10'
                s45path = path+'s45'
                s90path = path+'s90'
                s135path = path+'s135'
                s170path = path+'s170'
                for l in range(len(speed)):
                    if i%int(machine.architecture['data_split'][1]/100*len(speed)) == 0:
                        category = 'validation'
                    else:
                        category = 'train'
                    s10mat = preprocessing.normalize(h5py.File(s10path+speed[l],'r').get('acc'))
                    s45mat = preprocessing.normalize(h5py.File(s45path+speed[l],'r').get('acc'))
                    s90mat = preprocessing.normalize(h5py.File(s90path+speed[l],'r').get('acc'))
                    s135mat = preprocessing.normalize(h5py.File(s135path+speed[l],'r').get('acc'))
                    s170mat = preprocessing.normalize(h5py.File(s170path+speed[l],'r').get('acc'))
                    batchStack.update({
                    'batch'+str(l) :    Databatch([s10[1,:], s45[1,:], s90[1,:], s135[1,:], s170[1,:]],
                                        l,
                                        speed,
                                        element,
                                        category,
                                        damage_state = healthy[j])
                    })
                        
                
                NeuralNet.train_ad_hoc(machine, batchStack, epochs=200)
    else: 
        speed = sort(os.listdir(path +'s10'))
        s10path = path+'s10'
        s45path = path+'s45'
        s90path = path+'s90'
        s135path = path+'s135'
        s170path = path+'s170'
        for l in range(len(speed)):
            if i%int(machine.architecture['data_split'][1]/100*len(speed)) == 0:
                category = 'validation'
            else:
                category = 'train'
            s10mat = preprocessing.normalize(h5py.File(s10path+speed[l],'r').get('acc'))
            s45mat = preprocessing.normalize(h5py.File(s45path+speed[l],'r').get('acc'))
            s90mat = preprocessing.normalize(h5py.File(s90path+speed[l],'r').get('acc'))
            s135mat = preprocessing.normalize(h5py.File(s135path+speed[l],'r').get('acc'))
            s170mat = preprocessing.normalize(h5py.File(s170path+speed[l],'r').get('acc'))
            batchStack.update({
            'batch'+str(l) :    Databatch([s10[1,:], s45[1,:], s90[1,:], s135[1,:], s170[1,:]],
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

        

