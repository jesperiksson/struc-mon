#Other files and classes
from util import *
from Databatch import * 
# Modules
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential, Model, model_from_json
from tensorflow.python.keras.layers import Input, Dense, LSTM, concatenate, Activation, Reshape, Bidirectional
from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.python.keras import metrics, regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizers import RMSprop

class NeuralNet():
    def __init__(self,
                 arch,
                 name,
                 existing_model = False):

        self.arch = arch
        self.name = name
        self.target_sensor = self.arch['sensors'][self.arch['target_sensor']]
        self.pattern_sensors = self.arch['sensors'][self.arch['pattern_sensors'][0]]
        self.sensor_to_predict = arch['sensors'][arch['target_sensor']]
        if arch['early_stopping'] == True:
            self.early_stopping = [keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=0, 
                patience=arch['patience'],
                verbose=1,
                mode='auto',
                restore_best_weights=True)]

        else:
            self.early_stopping = None
        self.existing_model = existing_model
        self.n_sensors = len(arch['sensors'])    
        model_dict = {
            'test' : set_up_model_test(arch),
            '6' : set_up_model6(arch),
            '7' : set_up_model7(arch),
            '8' : set_up_model8(arch)
            }     
        if self.existing_model == False:
            model = model_dict[arch['model']]

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
        optimizer = keras.optimizers.Adam(
        learning_rate = arch['learning_rate'],
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False
        )
        model.compile(optimizer=optimizer, loss='mse', metrics=[rmse])
        plot_model(model, to_file='figs/'+name+'.png')
        model.summary()
        self.model = model
        self.score = None
        self.loss = [None]

    def train(self, series_stack):
        self.history = [None]
        self.loss = [None]
        self.val_loss = [None]
        print('\n Number of series being used for training:', len(series_stack[self.arch['preprocess_type']]), '\n')
        tic = time.time()
        for i in range(len(series_stack[self.arch['preprocess_type']])):
            series = series_stack[self.arch['preprocess_type']]['batch'+str(i%len(series_stack))]
            print(series)
            if series.category == 'train' or series.category == 'validation':
                print('\nFitting series: ', i, ' out of:', len(series_stack[self.arch['preprocess_type']]))
                X, Y = generator(self, series)
                patterns = {'speed_input' : np.array([series.normalized_speed])}
                for j in range(len(self.arch['active_sensors'])):
                    patterns.update({
                        'accel_input_'+str(self.arch['pattern_sensors'][j]) : X # Ange vilken
                    })
                targets = Y           
                history = self.model.fit(
                    x = X,#patterns,
                    y = Y,#targets, 
                    batch_size = self.arch['batch_size'],
                    epochs=self.arch['epochs'], 
                    verbose=1,
                    callbacks=self.early_stopping, #self.learning_rate_scheduler],
                    validation_split = self.arch['data_split']['validation']/100,
                    shuffle = True)
                self.history.append(history)
                self.loss.extend(history.history['loss'])
                self.val_loss.extend(history.history['val_loss'])  
                if self.arch['save_periodically'] == True and i % self.arch['save_interval'] == 0:
                    save_model(self.model,self.name)  
        self.model.summary()
        self.toc = np.round(time.time()-tic,1)
        print('Elapsed time: ', self.toc)
        return

    def evaluation(self, series_stack):
        for i in range(len(series_stack)):
            series = series_stack[self.arch['preprocess_type']]['batch'+str(i%len(series_stack))]
            steps = get_steps(self, series)
            if series.category == 'train':
                self.score = self.model.evaluate_generator(
                    generator_peak(
                        self, 
                        series),
                    steps = steps,
                    verbose = 1)
        print('Model score: ', self.model.metrics_names, self.score)
        return

    def evaluation_batch(self, series_stack):
        scores = []
        speeds = []
        for i in range(len(series_stack)):
            series = series_stack[self.arch['preprocess_type']]['batch'+str(i%len(series_stack))]
            if series.category == 'test':
                X, Y = generator(self, series)
                score = self.model.evaluate(
                    x = X,
                    y = Y,
                    batch_size = self.arch['batch_size'],
                    verbose = 1,
                    return_dict = True)
                speeds.extend([series.speed['km/h']])
                scores.extend([score['rmse']])
        results = {
            'scores' : scores[1:],
            'speeds' : speeds[1:],
            'steps' : len(speeds[1:]),
            'damage_state' : series.damage_state
        }
        return results

    def prediction(self, manual):
        series = manual['stack'][self.arch['preprocess_type']]['batch'+str(manual['series_to_predict']%len(manual['stack']))]
        steps = get_steps(self, series)
        X, Y = generator(self, series)
        predictions = self.model.predict(
            X,
            batch_size = self.arch['batch_size'],
            verbose = 1,
            steps = steps)
        indices = np.empty([steps, self.arch['n_target_steps']])
        for i in range(steps):
            start = i * self.arch['pattern_delta']+self.arch['n_pattern_steps']*self.arch['delta']
            end=i*self.arch['pattern_delta']+(self.arch['n_target_steps']+self.arch['n_pattern_steps'])*self.arch['delta']
            indices[i,:] = series.timesteps[start:end]
        prediction = {
            'prediction' : predictions,
            'hindsight' : Y,
            'steps' : steps,
            'indices' : indices
        }
        
        return prediction

    def forecast(machines, manual):
            # Machines
            machine_keys = list(machines.keys())
            # Shortcuts
            machine = machines[machine_keys[0]]
            delta = machine.arch['delta']
            n_pattern_steps = machine.arch['n_pattern_steps']
            n_target_steps = machine.arch['n_target_steps']
            # Series
            key = 'batch'+str(manual['series_to_predict']%len(manual['stack'][machine.arch['preprocess_type']]))
            n_steps = manual['stack'][machine.arch['preprocess_type']][key].n_steps
            series = manual['stack'][machine.arch['preprocess_type']][key]
            n_series = int((n_steps-n_pattern_steps)/n_target_steps)
            # Initial
            initial_indices = np.arange(0,delta*n_pattern_steps,delta)
            patterns = {}
            for i in range(len(machine.arch['pattern_sensors'])):
                patterns.update({ 
                'accel_input_'+machine.arch['pattern_sensors'][i] : 
                    np.reshape(
                        series.data[machine.sensor_to_predict][initial_indices], 
                        [1,machine.arch['n_pattern_steps']]
                    )
                })
            forecasts = patterns.copy()
            for i in range(n_series+1):
                old_patterns = patterns.copy()
                for j in range(len(machine_keys)):
                    machine = machines[machine_keys[j]] # Pick machine
                    prediction = machine.model.predict(
                        old_patterns,
                        batch_size = 1, 
                        verbose=0,
                        steps = 1) # Make prediction with machine
                    pattern = patterns['accel_input_'+machine.arch['pattern_sensors'][j]] # Extract pattern
                    pattern = np.delete(pattern,np.s_[0:n_target_steps:delta],1) # Remove first entty
                    pattern = np.hstack([pattern,prediction]) # Add prediciton last
                    patterns.update({
                        'accel_input_'+machine.arch['pattern_sensors'][j] : pattern
                        }) # Update patterns dicr
                    forecast = forecasts['accel_input_'+machine.arch['pattern_sensors'][j]] #Extract forecast
                    if i == n_series: # Edge case for last bit                   
                        forecast = np.hstack(
                            [forecast,
                            prediction[:,:n_steps%(n_series*n_target_steps+n_pattern_steps)]]
                            ) # Update forecast
                    else:
                        forecast = np.hstack([forecast,prediction]) # Update forecast
                    forecasts.update({
                        'accel_input_'+machine.arch['pattern_sensors'][j] : forecast
                        }) # Update forecasts dict
            return forecasts 

def generator(self, batch):
    '''
    Generator for when to use Peak accelerations and locations
    Each series of data is so big it has to be broken down
    '''
    steps = get_steps(self, batch)
    print(steps)
    X = np.empty([steps,self.arch['n_pattern_steps'], len(self.arch['pattern_sensors'])])
    Y = np.empty([steps,self.arch['n_target_steps']])
    for j in range(len(self.arch['pattern_sensors'])):     
        for k in range(steps):    
            pattern_start = k*self.arch['pattern_delta']
            pattern_finish = k*self.arch['pattern_delta']+self.arch['delta']*self.arch['n_pattern_steps']
            target_start = k*self.arch['pattern_delta']+self.arch['delta']*self.arch['n_pattern_steps'] # +1?
            target_finish = k*self.arch['pattern_delta']+self.arch['delta']*(self.arch['n_pattern_steps']+self.arch['n_target_steps'])
            X[k,:,j] = batch.data[j][pattern_start:pattern_finish]
            Y[k,:] = batch.data[j][target_start:target_finish]
    return X, Y
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
        k+self.arch['delta']*self.arch['n_pattern_steps'] + self.arch['delta']*self.arch['n_target_steps'],
        self.arch['delta'])
    peak_pattern = batch.peaks[self.arch['pattern_sensors'][j]][pattern_indices]
    location_pattern = batch.delta[self.arch['pattern_sensors'][j]][pattern_indices]
    peak_target = batch.peaks[self.arch['target_sensor']][target_indices]
    return peak_pattern, location_pattern, peak_target

def get_steps(self, series):
    steps = int(
        np.floor(
            (series.n_steps-(self.arch['n_pattern_steps']+self.arch['n_target_steps']))/self.arch['pattern_delta']
        )
    )
    return steps   

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
    accel_input = Input(
        shape=(
            #arch['batch_size'],
            arch['n_pattern_steps'], 
            1),
        name = 'accel_input_90')

    location_input = Input(
        shape=(
            #arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        name = 'location_input_90')

    merge_layer = concatenate([accel_input, location_input])
   
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
        stateful = False)(merge_layer)

    hidden_dense_3 = Dense(
        arch['n_units'][1],
        activation = arch['Dense_activation'],
        use_bias = True)(hidden_lstm_1)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_'+str(arch['sensors'][arch['target_sensor']]))(hidden_dense_3)
    model = Model(inputs = [accel_input, location_input], outputs = output, name='LSTM model')
    return model

######################################################################################################
def set_up_model_test(arch):
    input1 = Input(shape=(1,),name='1')
    input2 = Input(shape=(1,),name='2')
    conc = concatenate([input1,input2])
    output1 = Dense(1,name='out')(conc)
    model = Model(inputs = [input1,input2], outputs = output1)
    return model 
######################################################################################################
def set_up_model6(arch): # Vanilla 

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            1),
        name = 'accel_input_90')

    hidden_lstm_1 = LSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        activation = arch['LSTM_activation'],
        recurrent_activation = 'hard_sigmoid',
        use_bias = arch['bias'],
        dropout = 0.1,
        stateful = False)(accel_input)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_1)

    model = Model(inputs = accel_input, outputs = output)

    return model
######################################################################################################
def set_up_model7(arch):

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            1),
        name = 'accel_input_90')

    hidden_lstm_1 = LSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        activation = arch['LSTM_activation'],
        recurrent_activation = 'hard_sigmoid',
        use_bias = arch['bias'],
        dropout = 0.1,
        return_sequences = True,
        stateful = False)(accel_input)

    hidden_lstm_2 = LSTM(
        arch['n_units']['second'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        activation = arch['LSTM_activation'],
        recurrent_activation = 'hard_sigmoid',
        use_bias = arch['bias'],
        dropout = 0.1,
        stateful = False)(hidden_lstm_1)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_2)

    model = Model(inputs = accel_input, outputs = output)

    return model
######################################################################################################
def set_up_model8(arch): # Vanilla but bidirectional
    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            1),
        name = 'accel_input_90')

    hidden_lstm_1 = Bidirectional(LSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            1)),
        activation = arch['LSTM_activation'],
        recurrent_activation = 'hard_sigmoid',
        use_bias = arch['bias'],
        dropout = 0.1,
        stateful = False)(accel_input)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_1)

    model = Model(inputs = accel_input, outputs = output)

    return model
######################################################################################################

