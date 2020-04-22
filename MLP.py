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
                 architecture,
                 name,
                 batchStack,
                 early_stopping = True,
                 existing_model = False,
                 sensor_to_predict = 0):

        self.architecture = architecture
        self.ad_hoc_batchStack = None
        self.name = name
        self.model_type = self.architecture['model_type']
        self.activation = self.architecture['MLPactivation']
        self.sensor_to_predict = sensor_to_predict
        if early_stopping == True:
            self.early_stopping = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0, 
                                                 patience=3,
                                                 verbose=1,
                                                 mode='auto',
                                                 restore_best_weights=True)]
        else:
            self.early_stopping = None
        self.loss='mse'
        self.existing_model = existing_model
        n_sensors = len(architecture['sensors'])         
        if self.existing_model == False:
            model = set_up_model3(architecture)

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
        plot_model(model, to_file='model.png')
        model.summary()
        self.model = model
        self.score = None

    def train_measurements(self, batchStack, epochs = 200):
        '''
        Reshapes the data to the form 
        0 [x_00 = a_1, x_01 = a_1+delta, ..., x_0(n_pattern_steps) = a_(delta*n_pattern_steps)]
        1 [x_10 = a_2, x_11 = a_2+delta, ..., x_1(n_pattern_steps) = a_(delta*n_pattern_steps)]
        .
        .
        n_series  [x_n_series0 = a_n_series...]
        '''    
        self.history = self.model.fit_generator( generator(self,'train',batchStack), 
                                                 steps_per_epoch=100, 
                                                 epochs=epochs, 
                                                 verbose=1,
                                                 callbacks=self.early_stopping, 
                                                 validation_data = generator(self,'validation', batchStack),
                                                 validation_steps=5)
        self.model.summary()
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
        self.used_epochs = len(self.val_loss)
        return


  

  
    def evaluation(self, batchStack):
        self.score = self.model.evaluate_generator(generator(self, 'test', batchStack),
                                                   steps = self.architecture['data_split'][2]/100*len(batchStack),
                                                   verbose = 1)
        print('Model score: ', self.model.metrics_names, self.score)
        return

    def evaluation_batch(self, batchStack):
        scores = []
        for i in range(len(batchStack)):
            key = 'batch'+str(i%len(batchStack))
            if batchStack[key].category == 'test':
                patterns, targets = data_sequence(self, batchStack, key)
                speed = batchStack[key].speed
                score = self.model.test_on_batch(patterns, targets, reset_metrics=False)[1]#['loss', 'rmse']
                scores.extend([score,speed])
        return scores

    def prediction(self, batchStack, number):
        delta = self.architecture['delta']
        n_pattern_steps = self.architecture['n_pattern_steps']
        n_target_steps = self.architecture['n_target_steps']
        key = 'batch'+str(number%len(batchStack))
        n_series = int(batchStack[key].n_steps)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        if batchStack[key].category == 'test':
            for i in range(n_series):
                pattern_indices = np.arange(i,i+(delta)*n_pattern_steps,delta)
                target_indices = i+delta*n_pattern_steps
                patterns[i,:] = batchStack[key].data[self.sensor_to_predict][pattern_indices]
                targets[i,:] = batchStack[key].data[self.sensor_to_predict][target_indices]
            prediction = self.model.predict(patterns, batch_size=10, verbose=1)
            return(prediction, targets)
        else:
            print('\n Not a test batch \n')
        return

    def get_H_score(self, mse_threshold):
        right = 0
        wrong = 0
        delta = self.architecture['delta']
        n_pattern_steps = self.architecture['n_pattern_steps']
        n_target_steps = self.architecture['n_target_steps']
        n_series = n_series = int(batchStack[key].n_steps)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        for i in range(len(batchStack)):
            key = 'batch'+str(i%len(batchStack))
            batch = self.batchStack[key]
            if batch.category == 'validation':
                pred, target = self.prediction(i)
                mse = mean_squared_error(pred, target)
                if mse > mse_threshold:
                    wrong += 1
                else:
                    right +=1
            else:
                pass
            i += 1
        self.H_score = right/(wrong+right)
        return self.H_score
    
    def get_D_score(self, mse_threshold, batchStack):
        right = 0
        wrong = 0
        delta = self.architecture['delta']
        n_pattern_steps = self.architecture['n_pattern_steps']
        n_target_steps = self.architecture['n_target_steps']
        n_series = int(self.batchStack['batch1'].diff)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        for i in range(len(self.batchStack)):
            key = 'batch'+str(i%len(batchStack))
            batch = batchStack[key]
            if batch.category == 'validation':
                pred, target = self.prediction(i, True, batchStack)
                mse = mean_squared_error(pred, target)
                if mse > mse_threshold:
                    right += 1
                else:
                    wrong +=1
            else:
                pass
            i += 1
        self.D_score = right/(wrong+right)
        return self.D_score
# General Utilities
def data_sequence(self, batchStack, key):
    inputs = []#np.empty([n_sensors, n_pattern_steps])
    outputs = []#np.empty([n_sensors, n_target_steps])
    delta = self.architecture['delta']
    n_pattern_steps = self.architecture['n_pattern_steps']
    n_target_steps = self.architecture['n_target_steps']
    n_sensors = 3
    for j in range(n_sensors):
        n_series = int(batchStack[key].n_steps)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        for k in range(n_series):                
            pattern_indices = np.arange(j,j+(delta)*n_pattern_steps,delta)
            target_indices = j+delta*n_pattern_steps
            patterns[k,:] = batchStack[key].data[j][pattern_indices]
            targets[k,:] = batchStack[key].data[j][target_indices]
        inputs.append(patterns)
        outputs.append(targets)
    patterns = {'accel_input_half' : inputs[0],
                'accel_input_quarter' : inputs[1],
                'accel_input_third' : inputs[2],
                'speed_input' : np.repeat(np.array([batchStack[key].normalized_speed,]),n_series,axis=0)}
                
    targets = {'acceleration_output' : outputs[self.sensor_to_predict]}
    return patterns, targets

def generator(self, task, batchStack):    
    i = 0
    while True:
        key = 'batch'+str(i%len(batchStack))
        i+=1
        if batchStack[key].category == task:
            patterns, targets = data_sequence(self, batchStack, key)
            yield(patterns, targets)
        else:
            pass
    pass

def rmse(true, prediction):
    return backend.sqrt(backend.mean(backend.square(prediction - true), axis=-1))

####################################################################################################
def set_up_model1(architecture):

    accel_input_half = Input(shape=(architecture['n_pattern_steps'], ), name='accel_input_half')   
    accel_input_quarter = Input(shape=(architecture['n_pattern_steps'], ), name='accel_input_quarter')
    accel_input_third = Input(shape=(architecture['n_pattern_steps'], ), name='accel_input_third')
    speed_input = Input(shape=(1,), name='speed_input')

    s_half = Dense(architecture['n_units'][0],
                                activation = architecture['MLPactivation'],
                                use_bias=architecture['bias'])(accel_input_half)
    s_quarter = Dense(architecture['n_units'][0], 
                                    activation = architecture['MLPactivation'],
                                    use_bias=architecture['bias'])(accel_input_quarter)
    s_third = Dense(architecture['n_units'][0], 
                                    activation = architecture['MLPactivation'],
                                    use_bias=architecture['bias'])(accel_input_third)

    accels = concatenate([s_half, s_quarter, s_third])
    x = Dense(architecture['n_units'][1])(accels)
    x = Dense(architecture['n_target_steps'], activation='tanh')(x)
    speed = Dense(architecture['n_units'][1], activation = 'sigmoid')(speed_input)
    speed_accel = concatenate([x,speed])
    output = Dense(architecture['n_target_steps'], activation='tanh', name='acceleration_output')(speed_accel)  


    model = Model(inputs = [accel_input_half, accel_input_quarter, accel_input_third, speed_input], 
                  outputs = output)             
    return model
####################################################################################################
def set_up_model2(architecture):

    accel_input_10 = Input(shape=(architecture['n_pattern_steps'], ), name='accel_input_10')   
    accel_input_45 = Input(shape=(architecture['n_pattern_steps'], ), name='accel_input_45')
    accel_input_90 = Input(shape=(architecture['n_pattern_steps'], ), name='accel_input_90')
    accel_input_135 = Input(shape=(architecture['n_pattern_steps'], ), name='accel_input_135')
    accel_input_170 = Input(shape=(architecture['n_pattern_steps'], ), name='accel_input_170')

    s10 = Dense(architecture['n_units'][0], activation = architecture['MLPactivation'],use_bias=True)(accel_input_10)
    s45 = Dense(architecture['n_units'][0], activation = architecture['MLPactivation'],use_bias=True)(accel_input_45)
    s90 = Dense(architecture['n_units'][0], activation = architecture['MLPactivation'],use_bias=True)(accel_input_90)
    s135 = Dense(architecture['n_units'][0], activation = architecture['MLPactivation'],use_bias=True)(accel_input_135)
    s170 = Dense(architecture['n_units'][0], activation = architecture['MLPactivation'],use_bias=True)(accel_input_170)

    accels = concatenate([s10, s45, s90, s135, s170])
    speed_input = Input(shape=(1,1), name='speed_input')
    x = concatenate([accels, speed_input])
    x = Dense(architecture['n_units'][1])(x)

    output = Dense(architecture['n_target_steps'], activation='tanh', name='acceleration_output')(x)  


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
def set_up_model3(architecture):
    accel_input_half = Input(shape=(architecture['n_pattern_steps'], ), name='accel_input_half')
    s_half = Dense(architecture['n_units'][0],
                                activation = architecture['MLPactivation'],
                                use_bias=architecture['bias'])(accel_input_half)
    output = Dense(architecture['n_target_steps'], activation='tanh', name='acceleration_output')(s_half) 
    model = Model(inputs = accel_input_half, outputs = output)
    return model
