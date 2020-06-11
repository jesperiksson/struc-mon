# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats
import seaborn as sn
from sklearn.metrics import roc_curve, confusion_matrix, plot_confusion_matrix
import os
import h5py
import random
import pandas as pd

# Classes
from Databatch import *

''' Utilities for various classes'''

def fit_to_NN(
        data_split, 
        path, 
        healthy_percentage, 
        arch):
    
    types = arch['preprocess_type']
    paths = {}

    for i in range(len(arch['active_sensors'])):
        paths.update(
            {arch['active_sensors'][i] : path+'s'+arch['active_sensors'][i]+'/'})
    if arch['random_mode'] == 'debug':
        seed = 1
    elif arch['random_mode'] == 'test':
        seed = random.randint(0,10000)
    file_list = os.listdir(paths[arch['active_sensors'][0]])
    file_list.sort()
    random.Random(seed).shuffle(file_list)

    speeds = np.empty([len(file_list)])
    for i in range(len(file_list)):
        if len(file_list[i]) == 9:
            speeds[i] = int(file_list[i][0:5])
        elif len(file_list[i]) == 10:
            speeds[i] = int(file_list[i][0:6])
        else: 
            print('error')
    normalized_speeds = (speeds-min(speeds))/(max(speeds)-min(speeds))

    n_files = len(file_list)
    data_stack = {}
    preprocess_stack = {}
    peaks_stack = {}
    frequency_stack = {}
    extrema_stack = {}
    start = arch['from']
    to = arch['to']
    diff = to-start
    for i in range(n_files):
        data = [None]*len(paths)
        for j in range(len(paths)):
            mat = h5py.File(paths[arch['active_sensors'][j]] + file_list[i], 'r')
            data[j] = mat.get('acc')[1,start:to]

        if i/n_files < (data_split['train']/100):
            category = 'train'
        elif i/n_files > (data_split['validation']/100) and i/n_files < ((data_split['train']+data_split['validation'])/100):
            category = 'validation'
        else:
            category = 'test'
        if 'data' in types:
            data_stack.update({
                'batch'+str(i) : DataBatch(data,
                                 i,
                                 speeds[i]/1000,
                                 normalized_speeds[i],
                                 category,
                                 healthy_percentage)
                                })
        if 'frequency' in types:
            frequency_stack.update({
                'batch'+str(i) : frequencySpectrum(data,
                                 i,
                                 speeds[i]/1000,
                                 normalized_speeds[i],
                                 category,
                                 healthy_percentage)
                                })
        if 'peaks' in types:
            peaks_stack.update({
                'batch'+str(i) : peaks(data,
                                 i,
                                 speeds[i]/1000,
                                 normalized_speeds[i],
                                 category,
                                 healthy_percentage)
                                })
        if 'extrema' in types:
            extrema_stack.update({
            'batch'+str(i) : extrema(data,
                             i,
                             speeds[i]/1000,
                             normalized_speeds[i],
                             category,
                             healthy_percentage)
                            })

    preprocess_stack.update({
        'data' : data_stack,
        'frequency' : frequency_stack,
        'peaks' : peaks_stack,
        'extrema' : extrema_stack
        })    

    return preprocess_stack

def data_split_mode2(a):
    series_stack = {}
    damage_dir_list = os.listdir(a['path'])
    for j in range(len(damage_dir_list)):
        series_stack.update({
            damage_dir_list[j] : fit_to_NN(
                a['data_split'],
                a['path']+'/'+damage_dir_list[j]+'/', 
                int(damage_dir_list[j][:-1]),
                a)
            })
    return series_stack

def data_split_mode1(a):
    eval_series_stack = {}
    damage_dir_list = os.listdir(a['path'])
    for j in range(len(damage_dir_list)):
        if damage_dir_list[j] == '100%':
            eval_series_stack.update({
                damage_dir_list[j] : fit_to_NN(
                    a['data_split'],
                    a['path']+'/'+damage_dir_list[j]+'/',
                    int(damage_dir_list[j][:-1]),
                    a)
                })
        else:
            eval_series_stack.update({
                damage_dir_list[j] : fit_to_NN(
                    {'train' : 0, 'validation' : 0, 'test' : 100},
                    a['path']+'/'+damage_dir_list[j]+'/',
                    int(damage_dir_list[j][:-1]),
                    a)
                })
    return eval_series_stack

def save_model(model,name):
    model_json = model.to_json()
    with open('models/'+name+'.json', 'w') as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    model.save_weights('models/'+name+'.h5')
    print('Saved model:', name)

def plot_loss(self, name):
    plt.figure()
    plt.plot(range(len(self.loss)), self.loss, 'bo', label='Training loss', linewidth=0.3)
    plt.plot(range(len(self.val_loss)), self.val_loss, 'ro', label='Validation loss', linewidth=0.3)
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss - RMSE')
    plt.legend()
    plt.savefig(fname = name+'_loss_plot.png')
    plt.show() 

def plot_performance(score_stack, a, pod): # pod = prediction or forecast
    cmap = plt.cm.rainbow
    norm = colors.Normalize(vmin=33,vmax=100)
    percentage_keys = list(score_stack)
    for i in range(len(percentage_keys)): # Iterates over percentages
        for j in range(len(score_stack[percentage_keys[i]]['speeds'])):
            plt.plot(score_stack[percentage_keys[i]]['speeds'][j], 
                     score_stack[percentage_keys[i]]['scores'][j], 
                     color=cmap(norm(score_stack[percentage_keys[i]]['damage_state'][j])), 
                     marker='o')
    plt.xlabel('Speed [km/h]')
    plt.ylabel('Root Mean Square Error')
    plt.title('Sample scores for '+pod+' at sensor ' + str(a['target_sensor']))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm)
    cbar.set_label('Young\'s modulus percentage', rotation=270)
    plt.legend()
    plt.savefig(fname = a['name']+pod+'_performance_plot.png')
    plt.show()

def plot_confusion(prediction, name, pof):
    data = np.array(prediction['confusion_matrix'])
    text = np.asarray(
        [['True Negatives','False Positives'],
        ['False Negatives','True Positives']]
        )
    
    labels = (np.asarray(["{0}\n{1:.0f}".format(text,data) for text, data in zip(text.flatten(), data.flatten())])).reshape(2,2)
    sn.heatmap(
        prediction['confusion_matrix'], 
        annot = labels,
        fmt='', 
        cbar = False,
        cmap = 'binary')
    plt.savefig(fname = name+pof+'_confusion_matrix.png')
    plt.show()

def get_binary_prediction(score_stack, arch):
    damage_cases = list(score_stack.keys())
    phi = np.empty(1)
    labels = np.empty(1)
    mu = np.mean(score_stack['100%']['scores'])
    sigma = np.sqrt(np.var(score_stack['100%']['scores']))
    for i in range(len(damage_cases)):
        phi = np.append(phi,stats.norm.cdf(score_stack[damage_cases[i]]['scores'], mu, sigma),axis=0)
        if damage_cases[i] == '100%':
            labels = np.append(labels, np.zeros(len(score_stack[damage_cases[i]]['scores'])),axis=0)    
        else: 
            labels = np.append(labels, np.ones(len(score_stack[damage_cases[i]]['scores'])),axis=0) 
    prediction = np.heaviside(phi-arch['limit'], 1)
    prediction_dict = {
        'Phi' : phi[1:],
        'prediction' : prediction[1:],
        'labels' : labels[1:]}
    prediction_dict.update(
        {'confusion_matrix' : confusion_matrix(prediction_dict['prediction'],prediction_dict['labels'])}
        )
    return prediction_dict

def plot_roc(prediction):
    false_positive_rate, true_positive_rate = roc_curve(
        y_true = prediction['labels'],
        y_score = prediction['Phi']
        )
    plt.plot(false_positive_rate, true_positive_rate)
    plt.savefig(fname = name+'_roc.png')
    plt.show()

def plot_prediction(prediction, manual, net):
    plt.figure()
    
    if net == 'LSTM': 
        plt.plot(prediction['indices'], prediction['prediction'], 'b', linewidth=0.4)
        plt.plot(prediction['indices'], prediction['hindsight'], 'r', linewidth=0.4) 
        plt.legend(['Prediction', 'Signals'])   
    
    elif net == 'AELSTM':
        number = 3
        plt.plot(prediction['steps'], prediction['prediction'][number,:])
        plt.plot(prediction['steps'], prediction['hindsight'][number,:])
    elif net == 'MLP':
        plt.plot(prediction['indices'], prediction['prediction'], 'b', linewidth=0.4)
        plt.plot(prediction['indices'], prediction['hindsight'], 'r', linewidth=0.4) 
        plt.legend(['Prediction', 'Signals'])   
    plt.savefig(fname = net.name+'_prediction_plot.png')
    plt.show()
    return

def plot_forecast(forecast, manual, a):
    key = 'batch'+str(manual['series_to_predict'])
    forecast_keys = list(forecast.keys())
    for i in range(len(forecast_keys)): 
        plt.subplot(len(forecast_keys),1,i+1)
        plt.plot(
            manual['stack'][a['preprocess_type']][key].timesteps, 
            forecast[forecast_keys[i]][0], 
            'b', 
            linewidth=0.4)
        plt.plot(
            manual['stack'][a['preprocess_type']][key].timesteps, 
            manual['stack'][a['preprocess_type']][key].data[i], 
            'r', 
            linewidth=0.4)
        plt.xlabel('timesteps')
        plt.ylabel('accelerations')
        #plt.title('Forecast for response at '+str(manual['stack'][a['preprocess_type']][key].speed['km/h']+' km/h'))
        plt.legend(['Forecast', 'Signals']) 
    plt.savefig(fname = a['name']+'series'+str(manual['series_to_predict'])+'_forecast_plot.png')
    plt.show() 
                 









 
