#Other files and classes
from util import *
from Databatch import * 
# Modules
import time
import tensorflow as tf
import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, concatenate, Activation, Flatten 
from keras import metrics, callbacks, regularizers
from keras.utils import plot_model
from keras import backend
#from matplotlib import pyplot
from keras.optimizers import RMSprop

class NeuralNet():
    def __init__(self,
                 arch,
                 name,
                 existing_model = False):

        self.arch = arch
        self.name = name
        self.sensor_to_predict = arch['sensors'][arch['target_sensor']]
        if arch['early_stopping'] == True:
            self.early_stopping = [keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0, 
                patience=arch['patience'], # Pröva upp till typ 8
                verbose=1,
                mode='auto',
                restore_best_weights=True)]
        else:
            self.early_stopping = None
        self.loss='mse'
        self.existing_model = existing_model
        n_sensors = len(arch['sensors'])         
        if self.existing_model == False:
            model = set_up_model5(arch)

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
            amsgrad=False)
        model.compile(optimizer=optimizer, loss='mse', metrics=[rmse])
        plot_model(model, to_file='name.png')
        model.summary()
        self.model = model
        self.score = None

    def train(self, series_stack):
        '''
        Reshapes the data to the form 
        0 [x_00 = a_1, x_01 = a_1+delta, ..., x_0(n_pattern_steps) = a_(delta*n_pattern_steps)]
        1 [x_10 = a_2, x_11 = a_2+delta, ..., x_1(n_pattern_steps) = a_(delta*n_pattern_steps)]
        .
        .
        n_series  [x_n_series0 = a_n_series...]
        '''   
        tic = time.time() 
        self.history = self.model.fit(
            generator(
                self, 
                'train', 
                series_stack[self.arch['preprocess_type']]), 
            steps_per_epoch=16, 
            epochs=self.arch['epochs'], 
            verbose=1,
            callbacks=self.early_stopping, 
            validation_data = generator(self,'validation', series_stack[self.arch['preprocess_type']]),
            validation_steps=5)

        self.model.summary()
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
        self.used_epochs = len(self.val_loss)
        self.toc = np.round(time.time()-t,1)
        print('Elapsed time: ', self.toc)
        return

    def evaluation(self, series_stack): # Model score (loss and RMSE)
        self.score = self.model.evaluate_generator(
            generator(
                self, 
                'test', 
                series_stack[self.arch['preprocess_type']]),
            steps = self.arch['data_split']['test']/100*len(series_stack),
            verbose = 1)
        print('Model score: ', self.model.metrics_names, self.score)
        return

    def evaluation_batch(self, series_stack): # RMSE for a single batch
        scores = []
        speeds = []
        for i in range(len(series_stack[self.arch['preprocess_type']])):
            key = 'batch'+str(i%len(series_stack[self.arch['preprocess_type']]))
            series = series_stack[self.arch['preprocess_type']][key]
            if series.category == 'test':
                patterns, targets = data_sequence(self, series)
                speed = series.speed
                score = self.model.test_on_batch(patterns, targets, reset_metrics=False)[1]#['loss', 'rmse']
                speeds.extend([series.speed['km/h']])
                scores.extend([score])
                #print(scores)
        results = {
            'scores' : scores[1:],
            'speeds' : speeds[1:],
            'damage_state' : series.damage_state
        }
        return results

    def prediction(self, manual):
        delta = self.arch['delta']
        n_pattern_steps = self.arch['n_pattern_steps']
        n_target_steps = self.arch['n_target_steps']
        key = 'batch'+str(manual['series_to_predict']%len(manual['stack'][self.arch['preprocess_type']]))
        n_series = int(manual['stack'][self.arch['preprocess_type']][key].n_steps)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        indices = np.empty([n_series,1])        
        for i in range(n_series):
            pattern_indices = np.arange(i,i+(delta)*n_pattern_steps,delta)
            target_indices = i+delta*n_pattern_steps
            patterns[i,:] = manual['stack'][self.arch['preprocess_type']][key].data[self.sensor_to_predict][pattern_indices]
            targets[i,:] = manual['stack'][self.arch['preprocess_type']][key].data[self.sensor_to_predict][target_indices]
            indices[i,:] = i+delta*n_pattern_steps
        predictions = self.model.predict(patterns, batch_size=10, verbose=1)
        prediction = {
            'prediction' : predictions,
            'hindsight' : targets,
            'steps' : n_series,
            'indices' : indices,
            'sensor' : self.arch['target_sensor']
        }
        if manual['stack'][key].category == 'test':
            pass
        else:
            print('\n Not a test batch \n')
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
            series = manual['stack'][machine.arch['preprocess_type']][key]
            n_series = int(manual['stack'][machine.arch['preprocess_type']][key].n_steps)-int(n_pattern_steps)
            # Initial
            initial_indices = np.arange(0,delta*n_pattern_steps,delta)
            patterns = {}
            print(n_series, np.shape(series.data))
            for i in range(len(machine.arch['pattern_sensors'])):
                patterns.update({ 
                'accel_input_'+machine.arch['pattern_sensors'][i] : 
                    np.reshape(
                        series.data[machine.sensor_to_predict][initial_indices], 
                        [1,machine.arch['n_pattern_steps']]
                    )
                })
            forecasts = patterns.copy()
            for i in range(n_series):
                old_patterns = patterns.copy()
                for j in range(len(machine_keys)):
                    machine = machines[machine_keys[j]] # Pick machine
                    prediction = machine.model.predict(old_patterns, verbose=0) # Make prediction with machine
                    pattern = patterns['accel_input_'+machine.arch['pattern_sensors'][j]] # Extract pattern
                    pattern = np.delete(pattern,0,1) # Remove first enrty
                    pattern = np.hstack([pattern,prediction]) # Add prediciton last
                    patterns.update({
                        'accel_input_'+machine.arch['pattern_sensors'][j] : pattern
                        }) # Update patterns dicr
                    forecast = forecasts['accel_input_'+machine.arch['pattern_sensors'][j]] #Extract forecast
                    forecast = np.hstack([forecast,prediction]) # Update forecast
                    forecasts.update({
                        'accel_input_'+machine.arch['pattern_sensors'][j] : forecast
                        }) # Update forecasts dict
            return forecasts

# General Utilities
def data_sequence(self, series):
    inputs = []
    outputs = []
    delta = self.arch['delta']
    n_pattern_steps = self.arch['n_pattern_steps']
    n_target_steps = self.arch['n_target_steps']
    n_sensors = len(self.arch['sensors'])
    for j in range(n_sensors):
        n_series = int(series.n_steps)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        for k in range(n_series):                
            pattern_indices = np.arange(k,k+(delta)*n_pattern_steps,delta)
            target_indices = k+delta*n_pattern_steps
            patterns[k,:] = series.data[j][pattern_indices]
            targets[k,:] = series.data[j][target_indices]
        inputs.append(patterns)
        outputs.append(targets)
    print(np.shape(inputs))
    patterns = {'speed_input' : np.repeat(np.array([series.normalized_speed,]),n_series,axis=0)}
    for i in range(len(self.arch['pattern_sensors'])):
        patterns.update({'accel_input_'+self.arch['pattern_sensors'][i] : inputs[i]})
    targets = {'acceleration_output' : outputs[0]}
    return patterns, targets

def generator(self, task, series_stack):    
    i = 0
    while True:
        key = 'batch'+str(i%len(series_stack))
        i+=1
        if series_stack[key].category == task:
            patterns, targets = data_sequence(self, series_stack[key])
            yield(patterns, targets)
        else:
            pass
    pass

def rmse(true, prediction):
    return backend.sqrt(backend.mean(backend.square(prediction - true), axis=-1))



####################################################################################################
def set_up_model3(arch):
    
    accel_input = Input(
        shape=(arch['n_pattern_steps'], ),
        name='accel_input_'+str(arch['sensors'][0])
        )

    hidden_1 = Dense(arch['n_units']['first'],
                     activation = arch['Dense_activation'],
                     use_bias=arch['bias'])(accel_input)
    output = Dense(arch['n_target_steps'], activation='tanh', name='acceleration_output')(hidden_1) 
    model = Model(inputs = accel_input, outputs = output)
    return model
#######################################################################################################
def set_up_model4(arch):

    accel_input_45 = Input(
        shape = (arch['n_pattern_steps'], ),
        name='accel_input_45')

    accel_input_90 = Input(
        shape = (arch['n_pattern_steps'], ),
        name='accel_input_90')
    
    accel_input_135 = Input(
        shape = (arch['n_pattern_steps'], ),
        name='accel_input_135')

    accel_merge = concatenate([accel_input_45, accel_input_90, accel_input_135])

    hidden_1 = Dense(
        arch['n_units']['first'],
        activation = arch['Dense_activation'],
        use_bias=arch['bias'])(accel_merge)
    
    output = Dense(arch['n_target_steps'], activation='tanh', name='acceleration_output')(hidden_1)

    model = Model(inputs = [accel_input_45, accel_input_90, accel_input_135], outputs = output)
    return model 
####################################################################################################
def set_up_model5(arch):
    
    accel_input = Input(
        shape=(arch['n_pattern_steps'], ),
        name='accel_input_90')

    hidden_1 = Dense(arch['n_units']['first'],
                     activation = arch['Dense_activation'],
                     use_bias=arch['bias'])(accel_input)

    hidden_2 = Dense(arch['n_units']['second'],
                     activation = arch['Dense_activation'],
                     use_bias=arch['bias'])(hidden_1)

    output = Dense(arch['n_target_steps'], activation='tanh', name='acceleration_output')(hidden_2) 
    model = Model(inputs = accel_input, outputs = output)
    return model
