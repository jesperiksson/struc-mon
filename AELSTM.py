#Other files and classes
from util import *
from Databatch import * 
# Modules
import time
import tensorflow as tf
from tensorflow import keras
import keract
from tensorflow.python.keras.models import Sequential, Model, model_from_json
from tensorflow.python.keras.layers import Input, Dense, LSTM, concatenate, Activation, Reshape
from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.python.keras import metrics, regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import backend
#from matplotlib import pyplot
from tensorflow.python.keras.optimizers import RMSprop

class NeuralNet():
    def __init__(self,
                 arch,
                 name,
                 existing_model = False):

        self.arch = arch
        self.name = name
        self.target_sensor = self.arch['sensors'][self.arch['target_sensor']]
        #self.pattern_sensors = [None]*len([self.arch['pattern_sensors']])
        self.pattern_sensors = np.arange(0,len(self.arch['pattern_sensors']),1)
        self.sensor_to_predict = arch['sensors'][arch['target_sensor']]

        if arch['early_stopping'] == True:
            self.early_stopping = [keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=arch['min_delta'], 
                patience=arch['patience'],
                verbose=1,
                mode='auto',
                restore_best_weights=True)]

        else:
            self.early_stopping = None
        self.existing_model = existing_model
        self.n_sensors = len(arch['sensors'])    
        model_dict = {
            'AE' : set_up_model5(arch),
            }     
        if self.existing_model == False:
            model = model_dict[arch['model']]

        elif self.existing_model == True:
            model_path = 'models/'+self.arch['name']+'.json'
            weights_path = 'models/'+self.arch['name']+'.h5'
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
            amsgrad=False)
        model.compile(
            optimizer=optimizer, 
            loss=rmse, 
            metrics=[rmse])
        plot_model(model, to_file=name+'.png')
        model.summary()
        self.model = model
        self.score = None
        self.loss = [None]

    def train(self, series_stacks):
        tic = time.time()
        self.history = [None]
        self.loss = [None]
        self.val_loss = [None]
        keys = list(series_stacks.keys())
        for h in range(len(keys)):
            series_stack = series_stacks[keys[h]]
            print('\nTraining on ', keys[h],'% healthy data.\n')
            print('\n Number of series being used for training:', len(series_stack[self.arch['preprocess_type']]), '\n')
            for i in range(len(series_stack[self.arch['preprocess_type']])):
                series = series_stack[self.arch['preprocess_type']]['batch'+str(i)]
                if series.category == 'train' or series.category == 'validation':
                    print('\nFitting series: ', i, ' out of:', len(series_stack[self.arch['preprocess_type']]))
                    X, Y = generator(self, series)
                    patterns = {
                        'speed_input' : np.array([series.normalized_speed]),
                        'damage_input' : series.normalized_damage_state}
                    for j in range(len(self.arch['active_sensors'])):
                        patterns.update({
                            'accel_input_'+str(self.arch['pattern_sensors'][j]) : X # Ange vilken
                        })
                    targets = {
                        'accel_output_'+str(self.arch['pattern_sensors'][j]) : Y,
                        'damage_state_output' : series.normalized_damage_state
                        }           
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
        scores = []
        speeds = []
        damage_states = []
        for i in range(len(series_stack[self.arch['preprocess_type']])):
            series = series_stack[self.arch['preprocess_type']]['batch'+str(i)]
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
                damage_states.extend([series.damage_state])
            
        results = {
            'scores' : scores[:],
            'speeds' : speeds[:],
            'steps' : len(speeds[:]),
            'damage_state' : damage_states[:]
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
        indices = np.empty([steps, self.arch['in_out_put_size']])
        for i in range(steps):
            start = i * self.arch['pattern_delta']+self.arch['in_out_put_size']*self.arch['delta']
            end=i*self.arch['pattern_delta']+(self.arch['in_out_put_size']+self.arch['in_out_put_size'])*self.arch['delta']
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
        in_out_put_size = machine.arch['in_out_put_size']
        # Series
        key = 'batch'+str(manual['series_to_predict']%len(manual['stack'][machine.arch['preprocess_type']]))
        n_steps = manual['stack'][machine.arch['preprocess_type']][key].n_steps
        series = manual['stack'][machine.arch['preprocess_type']][key]
        n_series = int((n_steps-in_out_put_size)/in_out_put_size)
        # Initial
        initial_indices = np.arange(0,delta*in_out_put_size,delta)
        patterns = {}
        for i in range(len(machine.arch['pattern_sensors'])):
            patterns.update({ 
            'accel_input_'+machine.arch['pattern_sensors'][i] : 
                np.reshape(
                    series.data[machine.sensor_to_predict][initial_indices], 
                    [1,machine.arch['in_out_put_size']]
                )
            })
        forecasts = patterns.copy()
        evaluation = {}
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
                pattern = np.delete(pattern,np.s_[0:in_out_put_size:delta],1) # Remove first entty
                pattern = np.hstack([pattern,prediction]) # Add prediciton last
                patterns.update({
                    'accel_input_'+machine.arch['pattern_sensors'][j] : pattern
                    }) # Update patterns dict
                forecast = forecasts['accel_input_'+machine.arch['pattern_sensors'][j]] #Extract forecast
                if i == n_series: # Edge case for last bit                   
                    forecast = np.hstack(
                        [forecast,
                        prediction[:,:n_steps%(n_series*in_out_put_size+in_out_put_size)]]
                        ) # Update forecast
                else:
                    forecast = np.hstack([forecast,prediction]) # Update forecast
                forecasts.update({
                    'accel_input_'+machine.arch['pattern_sensors'][j] : forecast
                    }) # Update forecasts dict
        score = rmse_np(
            series.data[j][in_out_put_size:], 
            forecasts['accel_input_'+machine.arch['pattern_sensors'][j]][0][in_out_put_size:])
        speed = series.speed['km/h']
        damage_state = series.damage_state
        return forecasts, (score, speed, damage_state)

def generator(self, batch):
    '''
    Generator for when to use Peak accelerations and locations
    Each series of data is so big it has to be broken down
    '''
    steps = get_steps(self, batch)
    #print(steps)
    X = np.empty([steps,self.arch['in_out_put_size'], len(self.arch['pattern_sensors'])])
    Y = np.empty([steps,self.arch['in_out_put_size']])
    for j in range(len(self.arch['pattern_sensors'])):     
        for k in range(steps):    
            pattern_start = k*self.arch['pattern_delta']
            pattern_finish = k*self.arch['pattern_delta']+self.arch['delta']*self.arch['in_out_put_size']
            target_start = k*self.arch['pattern_delta']+self.arch['delta']*self.arch['in_out_put_size'] # +1?
            target_finish = k*self.arch['pattern_delta']+self.arch['delta']*(self.arch['in_out_put_size']+self.arch['in_out_put_size'])
            if self.arch['scheduled_sampling'] == False and k > 0:
                X[k,:,j] = batch.data[j][pattern_start:pattern_finish]
            elif self.arch['scheduled_sampling'] == True:
                rand = random.random()
                if rand < self.arch['sampling_rate']:
                    X[k,:,j] = self.model.predict(
                        x = X[k,:,j],
                        batch_size = None,
                        verbose = 0)
                elif rand >= self.arch['sampling_rate']:
                    Y[k,:] = batch.data[j][target_start:target_finish]
            Y[k,:] = batch.data[j][target_start:target_finish]
    return X, Y
    '''
    Generates inputs with shape [n_batches, n_timesteps, features]
    When there is not enough samples left to form a batch, the last batches will not be incorporated.
    TBD: in order to use several sensors, the sensor location needs to be included in the input
    '''

def get_steps(self, series):
    steps = int(
        np.floor(
            (series.n_steps-(self.arch['in_out_put_size']+self.arch['in_out_put_size']))/self.arch['pattern_delta']
        )
    )
    return steps   

def rmse(true, prediction):
    return backend.sqrt(backend.mean(backend.square(prediction - true), axis=-1))

def rmse_np(true, prediction):
    return np.sqrt(np.mean(np.square(prediction - true), axis=-1))

#######################################################################################################
def set_up_model5(arch):
    '''
    Peaks and their positions deltas as inputs
    '''
    peak_input = Input(
        shape=(
            arch['in_out_put_size'], 
            1),
        name = 'accel_input_90')
    '''
    speed_input = Input(
        shape=(
            1,),
        name = 'speed_input')
    '''
    encoded_1 = LSTM(
        arch['latent_dim']['first'],
        return_sequences = True)(peak_input)
    
    encoded_2 = LSTM(
        arch['latent_dim']['second'],
        return_sequences = True)(encoded_1)
    
    encoded_3 = LSTM(
        arch['latent_dim']['third'],
        return_sequences  = True)(encoded_2)

    decoded_3 = LSTM(
        arch['latent_dim']['third'],
        return_sequences = True)(encoded_3)

    decoded_2 = LSTM(
        arch['latent_dim']['second'], 
        return_sequences=True)(decoded_3)

    decoded_1 = LSTM(
        arch['latent_dim']['first'],
        return_sequences = True)(decoded_2)
   
    hidden_lstm_1 = LSTM(
        arch['n_units']['first'], 
        batch_input_shape = (
            arch['batch_size'],
            arch['in_out_put_size'],
            1),
        activation = arch['LSTM_activation'],
        recurrent_activation = 'hard_sigmoid',
        use_bias = arch['bias'],
        dropout = 0.1,
        stateful = False)(decoded_1)

    '''
    hidden_dense_1 = Dense(
        arch['n_units'][0],
        activation = arch['Dense_activation'],
        use_bias = True)(decoded_1)


    
    hidden_dense_2 = Dense(
        1,
        use_bias = True)(speed_input)
    '''
    hidden = LSTM(
        arch['in_out_put_size'], 
        activation='tanh',
        return_sequences = False, 
        name='peak_output_'+str(arch['sensors'][arch['target_sensor']]))(decoded_1)

    output = Reshape(
        input_shape = (0,arch['in_out_put_size'],1),
        target_shape = (arch['in_out_put_size'],1))(hidden)

    model = Model(inputs = peak_input, outputs = output)
    return model
######################################################################################################

