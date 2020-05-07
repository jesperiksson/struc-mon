#Other files and classes
from util import *
from Databatch import * 
# Modules
import tensorflow as tf
import keras
#import os
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, LSTM, concatenate, Activation, Reshape
from keras import metrics, callbacks, regularizers
from keras.utils import plot_model
from keras import backend
#from matplotlib import pyplot
from keras.optimizers import RMSprop

class NeuralNet():
    def __init__(self,
                 arch,
                 name,
                 early_stopping = True,
                 existing_model = False,
                 sensor_to_predict = 0):

        self.arch = arch
        self.name = name
        self.sensor_to_predict = sensor_to_predict
        if early_stopping == True:
            self.early_stopping = [keras.callbacks.EarlyStopping(monitor='loss',
                                                 min_delta=0, 
                                                 patience=1,
                                                 verbose=1,
                                                 mode='auto',
                                                 restore_best_weights=True)]
        else:
            self.early_stopping = None
        self.loss='mse'
        self.existing_model = existing_model
        self.n_sensors = len(arch['sensors'])         
        if self.existing_model == False:
            model = set_up_model4(arch)

        elif self.existing_model == True:
            model_path = 'models/'+self.name+'.json'
            weights_path = 'models/'+self.name+'.h5'
            json_file = open(model_path)
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(weights_path)
            model = loaded_model
            print('\n Loaded model: ', name)
        else:
            raise Error
        model.compile(optimizer='adam', loss='mse', metrics=[rmse])
        #plot_model(model, to_file='model.png')
        model.summary()
        self.model = model
        self.score = None
        self.loss = [None]

    def train_measurements(self, series_stack, epochs = 20):
        for i in range(len(series_stack)):
            key = 'batch'+str(i%len(series_stack))
            steps_per_epoch = int(
                np.floor(
                    series_stack[key].peak_steps/(self.arch['n_pattern_steps']+self.arch['n_target_steps'])
                )
            )
            print(steps_per_epoch, series_stack[key].peak_steps)
            epochs = 1
            if series_stack[key].category == 'train':
                '''
                steps_per_epoch is equal to the maximal number of patterns + targets that can fit in the 
                entire series.
                '''
                self.train_history = self.model.fit_generator(
                    generator_peak(
                        self,
                        series_stack[key]), 
                    steps_per_epoch=steps_per_epoch, # Number of batches to yield before performing backprop
                    epochs=epochs, # Enough to fit all samples in a series once
                    verbose=1,
                    callbacks=self.early_stopping)
                self.loss.append(self.train_history.history['loss'])
            elif series_stack[key].category == 'validation':
                self.val_history = self.model.evaluate_generator(
                    generator_peak(
                        self,
                        'validation',
                        series_stack[key]),
                    steps = steps_per_epoch)          
        self.model.summary()
        self.used_epochs = len(self.loss)   
        return

    def evaluation(self, series_stack):
        for i in range(len(series_stack)):
            key = 'batch'+str(i%len(series_stack))
            steps = int(
                np.floor(
                    series_stack[key].peak_steps/(self.arch['n_pattern_steps']+self.arch['n_target_steps'])
                )
            )
            if series_stack[key].category == 'train':
                self.score = self.model.evaluate_generator(
                    generator_peak(
                        self, 
                        series_stack[key]),
                    steps = steps,
                    verbose = 1)
        print('Model score: ', self.model.metrics_names, self.score)
        return

    def evaluation_batch(self, series_stack):
        scores = np.empty([len(series_stack), 3]) # [score, speed, batch]
        for i in range(len(series_stack)):
            key = 'batch'+str(i%len(series_stack))
            steps = int(np.floor(series_stack[key].n_steps/self.arch['n_pattern_steps']))
            if series_stack[key].category == 'test':
                score = self.model.evaluate_generator(
                    generator_peak(
                        self, 
                        series_stack[key]),
                    steps = steps, # Ska vara samma som steps_per_epoch i fit_generator
                    verbose = 1)
                speed = series_stack[key].speed
                scores[i,:] = [score[1], speed, int(i)]

        return scores

    def prediction(self, series_stack, number):
        key = 'batch'+str(number%len(series_stack))
        steps = int(
            np.floor(
                series_stack[key].peak_steps/(self.arch['n_pattern_steps']+self.arch['n_target_steps'])
            )
        )
        predictions = self.model.predict(
            generator_peak(
                self,
                series_stack[key]),
            verbose = 1,
            steps = steps)               
        #print(predictions)
        hindsight = np.empty([steps, self.arch['n_pattern_steps']])
        for i in range(steps):
            hindsight[i,:] = series_stack[number].peaks[i*self.arch['pattern_delta']:i*self.arch['pattern_delta']+self.arch['n_pattern_steps']]
        return predictions, hindsight

        return

    def modify_model(self):
        self.used_epochs = len(self.loss)  

def generator_accel(self, task, batch):
    '''
    Generator for when to use entire acceleration series
    '''    
    delta = self.arch['delta']
    n_pattern_steps = self.arch['n_pattern_steps']
    n_target_steps = self.arch['n_target_steps']
    n_sensors = len(self.arch['sensors'])
    n_series = int(batch.n_steps)-int(delta*n_pattern_steps)
    while True:
        for j in range(n_sensors):        
            pattern = np.empty([n_series,n_pattern_steps])
            target = np.empty([n_series,n_target_steps])
            for k in range(n_series):                
                pattern_indices = np.arange(k,k+(delta)*n_pattern_steps,delta)
                target_indices = k+delta*n_pattern_steps
                pattern[k,:] = batch.data[j][pattern_indices]
                target[k,:] = batch.data[j][target_indices]
                patterns = {'speed_input' :
                            np.repeat(np.array([batch.normalized_speed,]),n_series,axis=0),        
                            'accel_input_'+str(self.arch['sensors'][j]) : 
                            np.reshape(pattern,[n_series,n_pattern_steps,1])}
                targets = {'acceleration_output' : 
                            np.reshape(target,[n_series, n_target_steps])}
                yield(patterns, targets)

def generator_peak(self, batch):
    '''
    Generator for when to use Peak accelerations and locations
    Each series of data is so big it has to be broken down
    '''
    n_series = int(batch.n_steps)-int(self.arch['delta']*self.arch['n_pattern_steps'])
    while True:
        for j in range(len(self.arch['pattern_sensors'])):        
            for k in range(batch.n_steps):
                if k%self.arch['pattern_delta'] == 0 and batch.peak_steps >= k + self.arch['n_pattern_steps']+self.arch['n_target_steps']: # Filling the batch with samples               
                    peak_pattern, location_pattern, peak_target = add_pattern(self, j, k, batch)
                    patterns = {
                        'peak_input_'+str(self.arch['sensors'][self.arch['pattern_sensors'][j]]) : 
                         np.reshape(peak_pattern,[1,self.arch['n_pattern_steps'],1]),
                        'location_input_'+str(self.arch['sensors'][self.arch['pattern_sensors'][j]]) :
                        np.reshape(location_pattern,[1,self.arch['n_pattern_steps'],1]),
                        'speed_input' : 
                            np.repeat(np.array([batch.normalized_speed,]),n_series,axis=0)
                    }
                    targets = {
                        'peak_output_'+str(self.arch['sensors'][self.arch['target_sensor']]) : 
                            np.reshape(peak_target,[1, self.arch['n_target_steps']])
                    }
                    yield(patterns, targets)
                else: 
                    pass
    '''
    Generates inputs with shape [n_batches, n_timesteps, features]
    When there is not enough samples left to form a batch, the last batches will not be incorporated.
    TBD: in order to use several sensors, the sensor location needs to be included in the input
    '''

def add_pattern(self, j, k, batch):
    pattern_indices = np.arange(
        k, # start
        k+self.arch['delta']*self.arch['n_pattern_steps'], # end
        self.arch['delta']) # step
    target_indices = np.arange(
        k+self.arch['delta']*self.arch['n_pattern_steps'],
        k+self.arch['delta']*self.arch['n_pattern_steps'] + 
        self.arch['delta']*self.arch['n_target_steps'],
        self.arch['delta'])
    peak_pattern = batch.peaks[self.arch['pattern_sensors'][j]][pattern_indices]
    location_pattern = batch.peaks_indices[self.arch['pattern_sensors'][j]][pattern_indices]
    peak_target = batch.peaks[self.arch['target_sensor']][target_indices]
    return peak_pattern, location_pattern, peak_target
    

def rmse(true, prediction):
    return backend.sqrt(backend.mean(backend.square(prediction - true), axis=-1))


#######################################################################################################
def set_up_model3(arch):
    accel_input = Input(shape=(arch['n_pattern_steps'], 10),
                    batch_shape = (None, arch['n_pattern_steps'], 1),
                    name='accel_input_'+str(arch['sensors'][0]))
    hidden_1 = LSTM(arch['n_units'][0],
                    #input_shape = (arch['n_pattern_steps'], 1),
                    activation = arch['MLPactivation'],
                    recurrent_activation = 'hard_sigmoid',
                    use_bias=arch['bias'],
                    stateful = False)(accel_input)
    output = Dense(arch['n_target_steps'], activation='tanh', name='acceleration_output')(hidden_1)
    model = Model(inputs = accel_input, outputs = output)
    return model
#######################################################################################################
def set_up_model4(arch):
    '''
    Peaks and their positions deltas as inputs
    '''
    peak_input = Input(
        shape=(
            #arch['batch_size'],
            arch['n_pattern_steps'], 
            1),
        name = 'peak_input_90')

    location_input = Input(
        shape=(
            #arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        name = 'location_input_90')
   
    hidden_lstm_1 = LSTM(
        arch['n_units'][0],
        batch_input_shape = (
        arch['batch_size'],
        arch['n_pattern_steps'],
        1),
        activation = arch['LSTM_activation'],
        recurrent_activation = 'hard_sigmoid',
        use_bias = arch['bias'],
        dropout = 0.1,
        stateful = False)(peak_input)

    hidden_lstm_2 = LSTM(
        arch['n_units'][0],
        batch_input_shape = (
        arch['batch_size'],
        arch['n_pattern_steps'],
        1),
        activation = arch['LSTM_activation'],
        recurrent_activation = 'hard_sigmoid',
        use_bias = arch['bias'],
        dropout = 0.1,
        stateful = False)(location_input)

    hidden_dense_1 = Dense(
        arch['n_units'][0],
        activation = arch['Dense_activation'],
        use_bias = True)(hidden_lstm_1)
    '''
    hidden_dense_2 = Dense(
        arch['n_units'][0],
        activation = arch['Dense_activation'],
        use_bias = True)(location_input)
    '''
    '''
    hidden_dense_2 = Reshape(
        target_shape = (1, 1, arch['n_units'][0]),
        input_shape = (1, arch['n_units'][0]))(hidden_dense_2)
    '''

    merge_layer = concatenate([hidden_lstm_1, hidden_lstm_2])

    hidden_dense_3 = Dense(
        1,
        activation = arch['Dense_activation'],
        use_bias = True)(merge_layer)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_'+str(arch['sensors'][arch['target_sensor']]))(hidden_dense_1)
    model = Model(inputs = [peak_input, location_input], outputs = output)
    return model
