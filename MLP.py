#Other files and classes
from util import *
from Databatch import * 
# Modules
import tensorflow as tf
import keras
import os
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
                 series_stack,
                 early_stopping = True,
                 existing_model = False,
                 sensor_to_predict = 0):

        self.arch = arch
        self.name = name
        self.sensor_to_predict = sensor_to_predict
        if early_stopping == True:
            self.early_stopping = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0, 
                                                 patience=20, # Pröva upp till typ 8
                                                 verbose=1,
                                                 mode='auto',
                                                 restore_best_weights=True)]
        else:
            self.early_stopping = None
        self.loss='mse'
        self.existing_model = existing_model
        n_sensors = len(arch['sensors'])         
        if self.existing_model == False:
            model = set_up_model3(arch)

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
        plot_model(model, to_file='name.png')
        model.summary()
        self.model = model
        self.score = None

    def train_measurements(self, series_stack, epochs = 200):
        '''
        Reshapes the data to the form 
        0 [x_00 = a_1, x_01 = a_1+delta, ..., x_0(n_pattern_steps) = a_(delta*n_pattern_steps)]
        1 [x_10 = a_2, x_11 = a_2+delta, ..., x_1(n_pattern_steps) = a_(delta*n_pattern_steps)]
        .
        .
        n_series  [x_n_series0 = a_n_series...]
        '''    
        self.history = self.model.fit_generator( generator(self,'train',series_stack), 
                                                 steps_per_epoch=16, 
                                                 epochs=epochs, 
                                                 verbose=1,
                                                 callbacks=self.early_stopping, 
                                                 validation_data = generator(self,'validation', series_stack),
                                                 validation_steps=5)
        self.model.summary()
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
        self.used_epochs = len(self.val_loss)
        return

    def evaluation(self, series_stack): # Model score (loss and RMSE)
        self.score = self.model.evaluate_generator(generator(self, 'test', series_stack),
                                                   steps = self.arch['data_split']['test']/100*len(series_stack),
                                                   verbose = 1)
        print('Model score: ', self.model.metrics_names, self.score)
        return

    def evaluation_batch(self, series_stack): # RMSE for a single batch
        scores = []
        speeds = []
        for i in range(len(series_stack)):
            key = 'batch'+str(i%len(series_stack))
            if series_stack[key].category == 'test':
                patterns, targets = data_sequence(self, series_stack, key)
                speed = series_stack[key].speed
                score = self.model.test_on_batch(patterns, targets, reset_metrics=False)[1]#['loss', 'rmse']
                scores.extend([score])
                speeds.extend([speed])
                #print(scores)
        results = {
            'scores' : scores[1:],
            'speeds' : speeds[1:]
        }
        return results

    def prediction(self, manual):
        delta = self.arch['delta']
        n_pattern_steps = self.arch['n_pattern_steps']
        n_target_steps = self.arch['n_target_steps']
        key = 'batch'+str(manual['series_to_predict']%len(manual['stack']))
        n_series = int(manual['stack'][key].n_steps)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])        
        for i in range(n_series):
            pattern_indices = np.arange(i,i+(delta)*n_pattern_steps,delta)
            target_indices = i+delta*n_pattern_steps
            patterns[i,:] = manual['stack'][key].data[self.sensor_to_predict][pattern_indices]
            targets[i,:] = manual['stack'][key].data[self.sensor_to_predict][target_indices]
        predictions = self.model.predict(patterns, batch_size=10, verbose=1)
        print(np.shape(predictions))
        prediction = {
            'prediction' : predictions,
            'hindsight' : None,
            'steps' : 1
        }
        if manual['stack'][key].category == 'test':
            pass
        else:
            print('\n Not a test batch \n')
        return prediction



# General Utilities
def data_sequence(self, series_stack, key):
    inputs = []
    outputs = []
    delta = self.arch['delta']
    n_pattern_steps = self.arch['n_pattern_steps']
    n_target_steps = self.arch['n_target_steps']
    n_sensors = len(self.arch['sensors'])
    for j in range(n_sensors):
        n_series = int(series_stack[key].n_steps)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        for k in range(n_series):                
            pattern_indices = np.arange(k,k+(delta)*n_pattern_steps,delta)
            target_indices = k+delta*n_pattern_steps
            patterns[k,:] = series_stack[key].data[j][pattern_indices]
            targets[k,:] = series_stack[key].data[j][target_indices]
        inputs.append(patterns)
        outputs.append(targets)
    patterns = {'speed_input' : np.repeat(np.array([series_stack[key].normalized_speed,]),n_series,axis=0)}
    for i in range(n_sensors):
        patterns.update({'accel_input_'+str(self.arch['sensors'][i]) : inputs[i]})
    targets = {'acceleration_output' : outputs[0]}
    return patterns, targets

def generator(self, task, series_stack):    
    i = 0
    while True:
        key = 'batch'+str(i%len(series_stack))
        i+=1
        if series_stack[key].category == task:
            patterns, targets = data_sequence(self, series_stack, key)
            yield(patterns, targets)
        else:
            pass
    pass

def rmse(true, prediction):
    return backend.sqrt(backend.mean(backend.square(prediction - true), axis=-1))

####################################################################################################
def set_up_model1(arch):

    accel_input_half = Input(shape=(arch['n_pattern_steps'], ), name='accel_input_half')   
    accel_input_quarter = Input(shape=(arch['n_pattern_steps'], ), name='accel_input_quarter')
    accel_input_third = Input(shape=(arch['n_pattern_steps'], ), name='accel_input_third')
    speed_input = Input(shape=(1,), name='speed_input')

    s_half = Dense(arch['n_units'][0],
                                activation = arch['MLPactivation'],
                                use_bias=arch['bias'])(accel_input_half)
    s_quarter = Dense(arch['n_units'][0], 
                                    activation = arch['MLPactivation'],
                                    use_bias=arch['bias'])(accel_input_quarter)
    s_third = Dense(arch['n_units'][0], 
                                    activation = arch['MLPactivation'],
                                    use_bias=arch['bias'])(accel_input_third)

    accels = concatenate([s_half, s_quarter, s_third])
    x = Dense(arch['n_units'][1])(accels)
    x = Dense(arch['n_target_steps'], activation='tanh')(x)
    speed = Dense(arch['n_units'][1], activation = 'sigmoid')(speed_input)
    speed_accel = concatenate([x,speed])
    output = Dense(arch['n_target_steps'], activation='tanh', name='acceleration_output')(speed_accel)  


    model = Model(inputs = [accel_input_half, accel_input_quarter, accel_input_third, speed_input], 
                  outputs = output)             
    return model
####################################################################################################
def set_up_model2(arch):

    accel_input_10 = Input(shape=(arch['n_pattern_steps'], ), name='accel_input_10')   
    accel_input_45 = Input(shape=(arch['n_pattern_steps'], ), name='accel_input_45')
    accel_input_90 = Input(shape=(arch['n_pattern_steps'], ), name='accel_input_90')
    accel_input_135 = Input(shape=(arch['n_pattern_steps'], ), name='accel_input_135')
    accel_input_170 = Input(shape=(arch['n_pattern_steps'], ), name='accel_input_170')
    speed_input = Input(shape=(1,), name='speed_input')

    s10 = Dense(arch['n_units'][0], activation = arch['MLPactivation'],use_bias=True)(accel_input_10)
    s45 = Dense(arch['n_units'][0], activation = arch['MLPactivation'],use_bias=True)(accel_input_45)
    s90 = Dense(arch['n_units'][0], activation = arch['MLPactivation'],use_bias=True)(accel_input_90)
    s135 = Dense(arch['n_units'][0], activation = arch['MLPactivation'],use_bias=True)(accel_input_135)
    s170 = Dense(arch['n_units'][0], activation = arch['MLPactivation'],use_bias=True)(accel_input_170)

    accels = concatenate([s10, s45, s90, s135, s170])
    x = Dense(arch['n_units'][1])(accels)
    
    x = concatenate([accels, speed_input])
    x = Dense(arch['n_units'][1])(x)

    output = Dense(arch['n_target_steps'], activation='tanh', name='acceleration_output')(x)  


    model = Model(inputs=(  accel_input_10,
                            accel_input_45,
                            accel_input_90,
                            accel_input_135,
                            accel_input_170,
                            speed_input), 
            outputs = output)
    model.compile(optimizer='adam', loss='mse', metrics=['mse','acc'])
    model.summary()             
    return model

####################################################################################################
def set_up_model3(arch):
    
    accel_input = Input(shape=(arch['n_pattern_steps'], ),
                        name='accel_input_'+str(arch['sensors'][0]))
    hidden_1 = Dense(arch['n_units'][0],
                     activation = arch['Dense_activation'],
                     use_bias=arch['bias'])(accel_input)
    output = Dense(arch['n_target_steps'], activation='tanh', name='acceleration_output')(hidden_1) 
    model = Model(inputs = accel_input, outputs = output)
    return model
