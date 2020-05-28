# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats
from sklearn.metrics import roc_curve
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

    seed = 1000
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

    n_files = int(len(file_list)/1)
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

        if i/n_files <= data_split['train']/100:
            category = 'train'
        elif i/n_files > data_split['validation']/100 and i/n_files <= (data_split['train']+data_split['validation'])/100:
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

def get_eval_series(data_split, a, path):
    #element_dir_list = os.listdir(path)
    eval_series_stack = {}
    #for i in range(len(element_dir_list)):
    damage_dir_list = os.listdir(path)#+element_dir_list[i])
    for j in range(len(damage_dir_list)):
        eval_series_stack.update({
            damage_dir_list[j] : fit_to_NN(
                data_split,
                path+'/'+damage_dir_list[j]+'/', #+element_dir_list[i]
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

def plot_performance(score_stack, a):
    cmap = plt.cm.rainbow
    norm = colors.Normalize(vmin=33,vmax=100)
    percentage_keys = list(score_stack)
    
    for i in range(len(percentage_keys)): # Iterates over percentages
        for j in range(len(score_stack[percentage_keys[i]]['speeds'])):
            plt.plot(score_stack[percentage_keys[i]]['speeds'][j], 
                     score_stack[percentage_keys[i]]['scores'][j], 
                     color=cmap(norm(score_stack[percentage_keys[i]]['damage_state'])), 
                     marker='o')
    plt.xlabel('Speed [km/h]')
    plt.ylabel('Root Mean Square Error')
    plt.title('At sensor' + str(a['target_sensor']))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm)
    plt.legend()
    plt.show()

def score_evaluation(score_stack,a):

    limit=0.90                          #difference between healthy/unhealthy
    dmg_cases=[90, 81, 71, 62, 52, 43, 33]

    y_actual=[]    #error type 1
    y_predicted=[]    #error type 2
    
    
    scores=score_Stack[sensor_ind[0]]
    score=scores[100]
    
    data_set=score['scores']
    mu=np.mean(data_set)
    variance=np.var(data_set)
    sigma=np.sqrt(variance)

    ### HEALTHY EVALUATION ###
    norm_test=stats.norm.cdf((data_set-mu)/sigma)
    
    for i in range(len(norm_test)):
        y_actual.append(0)
        test_var=norm_test[i]
        if test_var > limit :
            y_predicted.append(1)
        else:
            y_predicted.append(0)
                    

    for k in range(len(dmg_cases)):
        X=scores[dmg_cases[k]]
        Xs=X['scores']
        norm_test_dmg=stats.norm.cdf((Xs-mu)/sigma)
        for j in range(len(norm_test_dmg)):
            test_var=norm_test_dmg[j]
            y_actual.append(1)
            if test_var < limit :
                y_predicted.append(0)
            else:
                y_predicted.append(1)
    raw_data={'Actual': y_actual,
          'Predicted': y_predicted
          }
    
    df = pd.DataFrame(raw_data, columns=['Actual','Predicted'])
    confusion_crosstab = pd.crosstab(df['Actual'], df['Predicted'],
                                   rownames=['Actual'], colnames=['Predicted'])
    #print(confusion_crosstab)
    data=np.array(confusion_crosstab)

    text=np.asarray([['True Negatives','False Positives'],
                     ['False Negatives','True Positives']])
    
    labels = (np.asarray(["{0}\n{1:.0f}".format(text,data) for text, data in zip(text.flatten(), data.flatten())])).reshape(2,2)
    sn.heatmap(confusion_crosstab, annot=labels, fmt='', cbar=False,cmap='binary')
    plt.show()
def get_binary_prediction(score_stack, arch, limit = 0.9):
    phi = [None] * len(score_stack)
    prediction = [None] * len(score_stack)
    for i in range(len(score_stack)):
        mu = np.mean(score_stack[arch['sensor_to_predict']][arch['healthy'][i]])
        sigma = np.sqrt(np.var(score_stack[arch['sensor_to_predict']][arch['healthy'][i]]))
        phi[i] = stats.norm.cdf(score_stack[arch['sensor_to_predict']][arch['healthy'][i]], mu, sigma)
        prediction[i] = np.heaviside(phi[i]-limit, 1)
    prediction = {
        'Phi' : phi,
        'prediction' : prediction}
    return prediction

def plot_roc(prediction):
    false_positive_rate, true_positive_rate = roc_curve(
        y_true = prediction['labels'],
        y_score = prediction['Phi']
        )
    plt.plot(false_positive_rate, true_positive_rate)
    plt.show()

def plot_prediction(prediction, manual, net):
    #key = 'batch'+str(manual['series_to_predict']%len(manual['stack']))
    #series = manual['stack'][key]
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
    plt.show() 
                 















 
