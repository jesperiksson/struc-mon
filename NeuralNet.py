"""# `NeuralNet` class
Most importantly contains the `self.model` attribute. The `model_dict` dictionary variable must be updated once a new model settingsitecture is added.
"""
from models import *

import time
import tensorflow as tf
'''
from tensorflow import keras
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import metrics, regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
'''

class NeuralNet():
    def __init__(
        self,
        settings,
        existing_model):

        self.settings = settings
        self.name = settings['name']
        #self.target_sensor = self.settings['sensors'][self.settings['target_sensor']]
        #self.pattern_sensors = self.settings['sensors'][self.settings['pattern_sensors'][0]]
        #self.sensor_to_predict = settings['sensors'][settings['target_sensor']]
        if settings['early_stopping'] == True:
            self.early_stopping = [keras.callbacks.EarlyStopping(
                monitor = settings['val_loss'],
                min_delta = settings['min_delta'], 
                patience = settings['patience'],
                verbose = 1,
                mode = 'auto',
                restore_best_weights = True)]

        else:
            self.early_stopping = None
        self.existing_model = existing_model
        #self.n_sensors = len(settings['sensors'])    
        model_dict = {
            'Single layer LSTM-CPU' : set_up_model6(settings),
            'Two layer CPU-LSTM' : set_up_model7(settings),
            'Single layer LSTM-non-CPU' : set_up_model1(settings), 
            'Two layer LSTM-non-CPU' : set_up_model2(settings),
            'Two layer LSTM CPU position' : set_up_model3(settings),
            'Single layer LSTM' : set_up_model8(settings),
            'Two layer LSTM' : set_up_model4(settings),
            'Three layer LSTM' : set_up_model9(settings),
            'Four layer LSTM' : set_up_model12(settings),
            'Single layer MLP' : set_up_model10(settings),
            'Two layer MLP' : set_up_model5(settings),
            'Three layer MLP' : set_up_model13(settings),
            'Four layer MLP' : set_up_model11(settings),
            'Single layer MLP dropout' : set_up_model14(settings),
            'Single layer LSTM dropout' : set_up_model15(settings)
            }     
        metric_dict = {
            'rmse' : [rmse],
            'mse' : 'mse',
            'val_loss' : 'val_loss',
            'mae' : 'mae'
            }
        if self.existing_model == False:
            model = model_dict[settings['model']]

        elif self.existing_model == True:
            model = load_model(settings)
        else:
            raise Error
        optimizer = keras.optimizers.Adam(
            learning_rate = settings['learning_rate'],
            beta_1 = 0.9,
            beta_2 = 0.999,
            epsilon = 1e-07,
            amsgrad = False)
        model.compile(
            optimizer = optimizer, 
            loss = settings['loss'],
            metrics = metric_dict[settings['metric']])
        '''
        fig = plot_model(
            model, 
            to_file = settings['model_plot_path'] + settings['model'] + '.png',
            show_shapes = False,
            #show_dtype = True,
            show_layer_names = False)
        '''
        model.summary()
        self.model = model
        #self.score = None
        #self.loss = [None]

    def train(self, data):
        tic = time.time()
        self.history = [None]
        self.loss = [None]
        self.val_loss = [None]
        
        X, Y = data_splitter(self, data, ['train', 'validation'])
        
        if np.shape(X)[0] == 0:
            pass
        else:         
            history = self.model.fit(
              x = X,#patterns,
              y = Y,#targets, 
              batch_size = self.settings['batch_size'],
              epochs=self.settings['epochs'], 
              verbose=1,
              callbacks=self.early_stopping, #self.learning_rate_scheduler],
              validation_split = self.settings['data_split']['validation']/100,
              shuffle = self.settings['shuffle'])
            self.history.append(history)
            self.loss.extend(history.history['loss'])
            self.val_loss.extend(history.history['val_loss'])  
            if self.settings['save_periodically'] == True and i % self.settings['save_interval'] == 0:
              save_model(self.model,self.settings)  
        self.model.summary()
        self.toc = np.round(time.time()-tic,1)
        print('Elapsed time: ', self.toc)
        return
        
    def read_and_train(self):
        if self.settings['data_function'] == 'one_by_one':
            to_learn = scan(learned = self.settings['learned'])
            for lesson in to_learn:
                data, learned = get_data_one_by_one(lesson)
                self.settings['learned'].update(learned)
                NeuralNet.train(model, data)
        elif model.settings['data_function'] == 'all_at_once':
            data = get_data_all_at_once(learned = self.settings['learned'])
            self.settings['learned'].update(learned)
            NeuralNet.train(model, data)
            
    def save_model(self):
        model_json = self.model.to_json()
        with open(self.settings['model_path'] + self.settings['fname'] + '.json', 'w') as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        self.model.save_weights(self.settings['model_path'] + self.settings['fname'] + '.h5')
        print('Saved model:', self.settings['fname'])
        
    def data_splitter(self, data_dict, categories):
        data_keys = list(data_dict.keys())
        for i in range(len(data_keys)):
            raw_data = data_dict[data_keys[i]].data
            print(raw_data[['x','y','z']][:-3])
            preprocessed_data = timeseries_dataset_from_array(
                data = raw_data[['x','y','z']][:-4],
                targets = raw_data[['x','y','z']][4:],
                sequence_length = self.settings['n_pattern_steps'],
                sequence_stride = self.settings['pattern_delta'],
                sampling_rate = self.settings['delta'],
                batch_size = 6
                )
        '''
        X = np.empty([0, self.arch['n_pattern_steps'], self.arch['features']])
        Y = np.empty([0, self.arch['n_target_steps']])
        for i in range(len(stack[self.arch['preprocess_type']])):
          series = stack[self.arch['preprocess_type']]['batch'+str(i)]
          if series.category in categories:
              steps = get_steps(self, series)
              x = np.empty([steps,self.arch['n_pattern_steps'], self.arch['features']])
              y = np.empty([steps,self.arch['n_target_steps']])
              for j in range(len(self.arch['pattern_sensors'])):    
                  for k in range(steps):    
                      pattern_start = k*self.arch['pattern_delta']
                      pattern_finish = k*self.arch['pattern_delta']+self.arch['delta']*self.arch['n_pattern_steps']
                      target_start = k*self.arch['pattern_delta']+self.arch['delta']*self.arch['n_pattern_steps']
                      target_finish = k*self.arch['pattern_delta']+self.arch['delta']*(self.arch['n_pattern_steps']+self.arch['n_target_steps'])
                      x[k,:,j] = series.data[j][pattern_start:pattern_finish]
                      if self.arch['pattern_sensors'][j] == self.arch['target_sensor']:
                          y[k,:] = series.data[j][target_start:target_finish]
                      if self.arch['positions'] == True:
                          x[k,:,-1] = np.arange(pattern_start, pattern_finish, self.arch['delta'])/biggest
              X = np.append(X,x,axis=0)
              Y = np.append(Y,y,axis=0)'''
        return preprocessed_data
     
def rmse(true, prediction):
    return backend.sqrt(backend.mean(backend.square(prediction - true), axis=-1))
def rmse_np(true, prediction):
    return np.sqrt(np.mean(np.square(prediction - true), axis=-1))

