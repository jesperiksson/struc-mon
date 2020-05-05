#Other files and classes
from util import *
from Databatch import * 
# Modules
import tensorflow as tf
import keras
#import os
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, LSTM, concatenate, Activation, Flatten 
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
        self.n_sensors = len(architecture['sensors'])         
        if self.existing_model == False:
            model = set_up_model4(architecture)

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

    def train_measurements(self, batchStack, epochs = 20):
        #tf.config.experimental.set_visible_devices([], 'GPU')
        #tf.config.experimental.set_memory_growth
        '''
        Reshapes the data to the form 
        0 [x_00 = a_1, x_01 = a_1+delta, ..., x_0(n_pattern_steps) = a_(delta*n_pattern_steps)]
        1 [x_10 = a_2, x_11 = a_2+delta, ..., x_1(n_pattern_steps) = a_(delta*n_pattern_steps)]
        .
        .
        n_series  [x_n_series0 = a_n_series...]
        '''
        for i in range(len(batchStack)):
            key = 'batch'+str(i%len(batchStack))
            steps = int(np.floor(batchStack[key].n_steps/self.architecture['n_pattern_steps']/epochs))
            if batchStack[key].category == 'train':
                '''
                Man väljer en train batch och tränar modellen på den. steps_per_epochs bör optimeras 
                så att man får ut max antal steg per ber batch, vilket beror på dess längd och batch
                size. 
                '''
                self.history = self.model.fit_generator(
                    generator_peak(
                        self,
                        'train',
                        batchStack[key]), 
                    steps_per_epoch=steps, 
                    epochs=epochs, 
                    verbose=1,
                    callbacks=self.early_stopping)

            elif batchStack[key].category == 'validation':
                self.history = self.model.evaluate_generator(
                    generator_peak(
                        self,
                        'validation',
                        batchStack[key]),
                    steps = steps)
                    
        print(self.history)
        self.model.summary()
        self.loss = self.history.history['loss']
        #self.val_loss = self.history.history['val_loss']
        self.used_epochs = len(self.loss)   
        return

    def evaluation(self, batchStack):
        for i in range(len(batchStack)):
            key = 'batch'+str(i%len(batchStack))
            steps = int(np.floor(batchStack[key].n_steps/self.architecture['n_pattern_steps']))
            if batchStack[key].category == 'train':
                self.score = self.model.evaluate_generator(
                    generator(
                        self, 
                        'test', 
                        batchStack[key]),
                    steps = steps,
                    verbose = 1)
        print('Model score: ', self.model.metrics_names, self.score)
        return

    def evaluation_batch(self, batchStack):
        scores = np.empty([len(batchStack), 3]) # [score, speed, batch]
        for i in range(len(batchStack)):
            key = 'batch'+str(i%len(batchStack))
            steps = int(np.floor(batchStack[key].n_steps/self.architecture['n_pattern_steps']))
            if batchStack[key].category == 'test':
                score = self.model.evaluate_generator(generator(self, 'test', batchStack[key]),
                                           steps = steps, # Ska vara samma som steps_per_epoch i fit_generator
                                           verbose = 1)
                speed = batchStack[key].speed
                scores[i,:] = [score[1], speed, int(i)]

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

    def modify_model(self):
        self.used_epochs = len(self.loss)  
    

def data_sequence(self, batchStack, key):
    inputs = []
    outputs = []
    delta = self.architecture['delta']
    n_pattern_steps = self.architecture['n_pattern_steps']
    n_target_steps = self.architecture['n_target_steps']
    n_sensors = len(self.architecture['sensors'])
    n_series = int(batchStack[key].n_steps)-int(delta*n_pattern_steps)
    for j in range(n_sensors):        
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        for k in range(n_series):                
            pattern_indices = np.arange(k,k+(delta)*n_pattern_steps,delta)
            target_indices = k+delta*n_pattern_steps
            patterns[k,:] = batchStack[key].data[j][pattern_indices]
            targets[k,:] = batchStack[key].data[j][target_indices]
        inputs.append(patterns)
        outputs.append(targets)
    patterns = {'speed_input' : np.repeat(np.array([batchStack[key].normalized_speed,]),n_series,axis=0)}
    for i in range(n_sensors):
        patterns.update(
        {'accel_input_'+str(self.architecture['sensors'][i]) : np.reshape(inputs[i],[1, n_series,n_pattern_steps])}
                        )
    targets = {'acceleration_output' : outputs[0]}
    print(np.shape(patterns['accel_input_90']), np.shape(targets['acceleration_output']))
    return patterns, targets

def generator_accel(self, task, batch):
    '''
    Generator for when to use entire acceleration series
    '''    
    delta = self.architecture['delta']
    n_pattern_steps = self.architecture['n_pattern_steps']
    n_target_steps = self.architecture['n_target_steps']
    n_sensors = len(self.architecture['sensors'])
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
                            'accel_input_'+str(self.architecture['sensors'][j]) : 
                            np.reshape(pattern,[n_series,n_pattern_steps,1])}
                targets = {'acceleration_output' : 
                            np.reshape(target,[n_series, n_target_steps])}
                yield(patterns, targets)

def generator_peak(self, task, batch):
    '''
    Generator for when to use Peak accelerations and locations
    '''
    delta = self.architecture['delta']
    n_pattern_steps = self.architecture['n_pattern_steps']
    n_target_steps = self.architecture['n_target_steps']
    n_sensors = len(self.architecture['sensors'])
    n_series = int(batch.n_steps)-int(delta*n_pattern_steps)
    while True:
        for j in range(n_sensors):        
            peak_pattern = np.empty([n_series,n_pattern_steps])
            location_pattern = np.empty([n_series,n_pattern_steps])
            peak_target = np.empty([n_series,n_target_steps])
            for k in range(n_series):                
                pattern_indices = np.arange(k,k+(delta)*n_pattern_steps,delta)
                target_indices = k+delta*n_pattern_steps
                peak_pattern[k,:] = batch.peaks[j][pattern_indices]
                location_pattern[k,:] = batch.locations[j][pattern_indices]
                peak_targets[k,:] = batch.peaks[j][target_indices]
                patterns = {
                    'peak_input' : np.reshape(peak_pattern[2],[n_series,n_pattern_steps,1]),
                    'location_input' : np.reshape(location_pattern,[n_series,n_pattern_steps,1]),
                    'speed_input' : np.repeat(np.array([batch.normalized_speed,]),n_series,axis=0)}
                targets = {
                    'peak_output' : np.reshape(peak_target,[n_series, n_target_steps])}
                print(patterns)
                yield(patterns, targets)

def rmse(true, prediction):
    return backend.sqrt(backend.mean(backend.square(prediction - true), axis=-1))

####################################################################################################
def set_up_model1(architecture):
    n_batches = None # i.e. varying
    n_sensors = len(architecture['sensors'])
    n_steps = architecture['n_pattern_steps']
    accel_input = Input(shape=(n_batches, n_steps, n_sensors), name='accel_input')   
    speed_input = Input(shape=(1,), name='speed_input')

    s = LSTM(architecture['n_units'][0],
             activation = architecture['LSTMactivation'],
             use_bias=architecture['bias'])(accel_input)

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
#######################################################################################################
def set_up_model3(architecture):
    accel_input = Input(shape=(architecture['n_pattern_steps'], 10),
                    batch_shape = (None, architecture['n_pattern_steps'], 1),
                    name='accel_input_'+str(architecture['sensors'][0]))
    hidden_1 = LSTM(architecture['n_units'][0],
                    #input_shape = (architecture['n_pattern_steps'], 1),
                    activation = architecture['MLPactivation'],
                    recurrent_activation = 'hard_sigmoid',
                    use_bias=architecture['bias'],
                    stateful = False)(accel_input)
    output = Dense(architecture['n_target_steps'], activation='tanh', name='acceleration_output')(hidden_1)
    model = Model(inputs = accel_input, outputs = output)
    return model
#######################################################################################################
def set_up_model4(architecture):
    '''
    Peaks and their positions deltas as inputs
    '''
    peak_input = Input(
        shape=(
            architecture['n_pattern_steps'], 
            10),
        batch_shape=(
            None, 
            architecture['n_pattern_steps'], 
            1),
        name='peak_input_'+str(architecture['sensors'][0]))
    location_input = Input(
        shape=(
            architecture['n_pattern_steps'],
            10),
        batch_shape=(
            None,
            architecture['n_pattern_steps'],
            1),
        name = 'location_input'+str(architecture['sensors'][0]))

    hidden_lstm_1 = LSTM(
        architecture['n_units'][0],
        activation = architecture['LSTM_activation'],
        recurrent_activation = 'hard_sigmoid',
        use_bias = architecture['bias'],
        stateful = False)(peak_input)

    hidden_dense_1 = Dense(
        architecture['n_units'][0],
        activation = architecture['Dense_activation'],
        use_bias = True)(location_input)

    merge_layer = concatenate([hidden_lstm_1, hidden_dense_1])

    hidden_dense_2 = Dense(
        architecture['n_units'][1],
        activation = architecture['Dense_activation'],
        use_bias = True)(merge_layer)

    output = Dense(architecture['n_target_steps'], activation='tanh', name='peak_output')(hidden_dense_2)
    model = Model(inputs = [peak_input, location_input], outputs = output)
    return model
