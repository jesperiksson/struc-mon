# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.stats as stats
from sklearn.metrics import roc_curve
import os
import h5py
import random

# Classes

''' Utilities for various classes'''

def fit_to_NN_ad_hoc(data_split, path, damaged_element, healthy_percentage, arch):
    from Databatch import DataBatch
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
    normalized_speeds = (speeds-min(speeds))/(max(speeds)-min(speeds))

    n_files = int(len(file_list)/1)
    series_stack = {}
    start = arch['from']
    to = arch['to']
    diff = to-start
    for i in range(n_files):
        data = [None]*len(paths)
        for j in range(len(paths)):
            mat = h5py.File(paths[arch['active_sensors'][j]] + file_list[i], 'r')
            data[j] = mat.get('acc')[1,start:to]
        '''
        try:
            s10mat = h5py.File(paths['s10path'] + file_list[i],'r')
            data[0] = s10mat.get('acc')
        except OSError:
            data[0] = np.zeros(diff)
        try:
            s45mat = h5py.File(paths['s45path'] + file_list[i],'r')
            data[1] = s45mat.get('acc')
        except OSError:
            data[1] = np.zeros(diff)
        try:
            s90mat = h5py.File(paths['s90path'] + file_list[i],'r')
            data[2] = s90mat.get('acc')
        except OSError:
            data[2] = np.zeros(diff)
        try:
            s135mat = h5py.File(paths['s135path'] + file_list[i],'r')
            data[3] = s135mat.get('acc')
        except OSError:
            data[3] = np.zeros(diff)
        try:
            s170mat = h5py.File(paths['s170path'] + file_list[i],'r')
            data[4] = s170mat.get('acc')
        except OSError:
            data[4] = np.zeros(diff)        
        #speed = int(file_list[i][0:5])/1000
        '''

        if i/n_files <= data_split['train']/100:
            category = 'train'
        elif i/n_files > data_split['validation']/100 and i/n_files <= (data_split['train']+data_split['validation'])/100:
            category = 'validation'
        else:
            category = 'test'
        series_stack.update({
            'batch'+str(i) : DataBatch(data,
                             i,
                             speeds[i]/1000,
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
            plt.plot(scoreStack[percentage_keys[j]]['speeds'], 
                     scoreStack[percentage_keys[j]]['scores'], 
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
        
def plot_prediction(prediction, manual, net):
    key = 'batch'+str(manual['series_to_predict']%len(manual['stack']))
    series = manual['stack'][key]
    plt.figure()
    if net == 'LSTM' or net == 'AELSTM':
        for i in range(prediction['steps']):
            plt.plot(prediction['indices'][i,:], prediction['prediction'][i,:], 'b', linewidth=0.4)
            plt.plot(prediction['indices'][i,:], prediction['hindsight'][i,:], 'r', linewidth=0.4)    
        plt.plot(series.timesteps, series.data[prediction['sensor']], 'g', linewidth=0.05)
        plt.legend(['Prediction','Data','Signals'])
    elif net == 'MLP':
        plt.plot(prediction['indices'], prediction['prediction'], 'b', linewidth=0.4)
        plt.plot(prediction['indices'], prediction['hindsight'], 'r', linewidth=0.4) 
        plt.legend(['Prediction', 'Signals'])   
    
    plt.show()
    return

def get_eval_series(data_split, damaged_element, a):
    eval_series_stack = {
        '90%' : fit_to_NN_ad_hoc(
            data_split,
            'our_measurements/e'+str(damaged_element)+'/90%/',
            damaged_element,
            90,
            a
        ),
        '81%' : fit_to_NN_ad_hoc(
            data_split,
            'our_measurements/e'+str(damaged_element)+'/81%/',
            damaged_element,
            81,
            a
        ),
        '71%' : fit_to_NN_ad_hoc(
            data_split,
            'our_measurements/e'+str(damaged_element)+'/71%/',
            damaged_element,
            71,
            a
        ),
        '62%' : fit_to_NN_ad_hoc(
            data_split,
            'our_measurements/e'+str(damaged_element)+'/62%/',
            damaged_element,
            62,
            a
        ),     
        '52%' : fit_to_NN_ad_hoc(
            data_split,
            'our_measurements/e'+str(damaged_element)+'/52%/',
            damaged_element,
            52,
            a
        ),
        '43%' : fit_to_NN_ad_hoc(
            data_split,
            'our_measurements/e'+str(damaged_element)+'/43%/',
            damaged_element,
            43,
            a
        ),
        '33%' : fit_to_NN_ad_hoc(
            data_split,
            'our_measurements/e'+str(damaged_element)+'/33%/',
            damaged_element,
            33,
            a
        )
    }
    return eval_series_stack



def score_evaluation(score_stack,a,):
    limit=0.90                          #difference between healthy/unhealthy
    dmg_cases=[90, 81, 71, 62, 52, 43, 33]

    error1=0    #error type 1
    error2=0    #error type 2
    
    
    scores=score_stack[sensor_ind[0]]
    score=scores[100]
    
    data_set=score['scores']
    mu=np.mean(data_set)
    variance=np.var(data_set)
    sigma=np.sqrt(variance)

    ### HEALTHY EVALUATION ###
    norm_test=stats.norm.cdf((data_set-mu)/sigma)
    
    for i in range(len(norm_test)):
        if norm_test[i] > limit :
           error1+= 1
                    

    print('False positives ' + str(error1) + ' out of ' + str(len(norm_test)))

    tests=0
    for k in range(len(dmg_cases)):
        X=scores[dmg_cases[k]]
        Xs=X['scores']
        norm_test_dmg=stats.norm.cdf((Xs-mu)/sigma)
        for j in range(len(norm_test_dmg)):
            test_var=norm_test_dmg[j]
            tests+= 1
            if test_var < limit :
                error2+= 1
                print(dmg_cases[k])

    print('False negatives ' + str(error2) + ' out of ' + str(tests)) 

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
                  
