#Other files and classes
from util import *
from Databatch import * 
# Modules
import tensorflow as tf
import keras
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
        self.target_sensor = self.arch['sensors'][self.arch['target_sensor']]
        self.pattern_sensors = self.arch['sensors'][self.arch['pattern_sensors']]
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
        amsgrad=False
        )
        model.compile(optimizer=optimizer, loss='mse', metrics=[rmse])
        plot_model(model, to_file=name+'.png')
        model.summary()
        self.model = model
        self.score = None
        self.loss = [None]

    def train_measurements(self, series_stack, epochs = 1):
        self.history = [None]
        self.loss = [None]
        self.val_loss = [None]
        for i in range(len(series_stack)):
            series = series_stack[self.arch['preprocess_type']]['batch'+str(i%len(series_stack))]
            steps = get_steps(self, series)
            try:
                indices = np.random.choice(
                    steps,
                    steps,
                    replace = False
                )
            except ValueError:
                print('ValueError')
                continue
            split = int(np.ceil(self.arch['data_split']['validation']/100*steps))
            exclude_validation = indices[split:] # To be excluded from validation, i.e. included in training
            exclude_train = indices[:split] # To be excluded from training
            train_steps = len(exclude_validation)
            validation_steps = len(exclude_train)
            print(validation_steps)
            if series.category == 'train' or series.category == 'validation':
                '''
                steps_per_epoch is equal to the maximal number of patterns + targets that can fit in the 
                entire series.
                '''
                history = self.model.fit_generator(
                    generator_peak(
                        self,
                        series,
                        exclude_train), 
                    steps_per_epoch=train_steps, # Number of batches to yield before performing backprop
                    epochs=self.arch['epochs'], # Enough to fit all samples in a series once
                    verbose=1,
                    callbacks=self.early_stopping,
                    validation_data = generator_peak(
                        self,
                        series,
                        exclude_validation),
                    validation_steps=validation_steps)
                self.history.append(history)
                self.loss.extend(history.history['loss'])
                self.val_loss.extend(history.history['val_loss'])  
        self.model.summary()
        #self.used_epochs = len(self.loss)   
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
            steps = int(np.floor(series.n_steps/self.arch['n_pattern_steps']))
            if series.category == 'test':
                score = self.model.evaluate_generator(
                    generator_peak(
                        self, 
                        series),
                    steps = steps, # Ska vara samma som steps_per_epoch i fit_generator
                    verbose = 1)
                speeds.extend([series.speed])
                scores.extend([score])
        results = {
            'scores' : scores[1:],
            'speeds' : speeds[1:]
        }

        return results

    def prediction(self, manual):
        series = manual[self.arch['preprocess_type']]['stack']['batch'+str(manual['series_to_predict']%len(manual['stack']))]
        steps = get_steps(self, series)
        predictions = self.model.predict(
            generator_peak(
                self,
                series),
            verbose = 1,
            steps = steps)               
        indices = np.empty([steps, self.arch['n_target_steps']])
        hindsight = np.empty([steps, self.arch['n_target_steps']])
        for i in range(steps):
            start = i * self.arch['pattern_delta'] + self.arch['n_pattern_steps']*self.arch['delta']
            end = i * self.arch['pattern_delta'] + (self.arch['n_target_steps'] + self.arch['n_pattern_steps'])*self.arch['delta'] 

            indices[i,:] = series.indices[self.arch['target_sensor']][start:end]
            hindsight[i,:] = series.peaks[self.arch['target_sensor']][start:end]
        prediction = {
            'prediction' : predictions,
            'indices' : indices,
            'hindsight' : hindsight,
            'steps' : steps,
            'sensor' : self.arch['target_sensor']
        }
        return prediction

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

def generator_peak(self, batch, include = ['foobar']):
    '''
    Generator for when to use Peak accelerations and locations
    Each series of data is so big it has to be broken down
    '''
    while True:
        for j in range(len(self.arch['pattern_sensors'])):
            l = 0        
            for k in range(batch.steps[self.arch['target_sensor']]):
                if k%self.arch['pattern_delta'] == 0 and batch.steps[self.arch['pattern_sensors'][j]] >= k + (self.arch['n_pattern_steps']+self.arch['n_target_steps'])*self.arch['delta']: # Filling the batch with samples
                    if l not in include:            
                        peak_pattern, location_pattern, peak_target = add_pattern(self, j, k, batch)
                        patterns = {
                            'peak_input_'+str(self.arch['sensors'][self.arch['pattern_sensors'][j]]) : 
                             np.reshape(peak_pattern,[1,self.arch['n_pattern_steps'],1]),
                            'location_input_'+str(self.arch['sensors'][self.arch['pattern_sensors'][j]]) :
                            np.reshape(location_pattern,[1,self.arch['n_pattern_steps'],1]),
                            'speed_input' : np.array([batch.normalized_speed]),
                            'index_input' : np.array([k / batch.steps[self.arch['target_sensor']]])
                        }
                        targets = {
                            'peak_output_'+str(self.arch['sensors'][self.arch['target_sensor']]) : 
                            np.reshape(peak_target,[1, self.arch['n_target_steps']])
                        }
                        yield(patterns, targets)
                    l+=1 
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
        k+self.arch['delta']*self.arch['n_pattern_steps'] + self.arch['delta']*self.arch['n_target_steps'],
        self.arch['delta'])
    peak_pattern = batch.peaks[self.arch['pattern_sensors'][j]][pattern_indices]
    location_pattern = batch.delta[self.arch['pattern_sensors'][j]][pattern_indices]
    peak_target = batch.peaks[self.arch['target_sensor']][target_indices]
    return peak_pattern, location_pattern, peak_target

def get_steps(self, series):
    steps = int(
        np.floor(
            (series.steps[self.arch['target_sensor']]-(self.arch['n_pattern_steps']+self.arch['n_target_steps']))/self.arch['pattern_delta']
        )
    )
    if steps < 0:
        print(series.batch_num)
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

    merge_layer = concatenate([peak_input, location_input])
   
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
    model = Model(inputs = [peak_input, location_input], outputs = output, name='LSTM model')
    return model
#######################################################################################################
def set_up_model5(arch):
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

    index_input = Input(
        shape=(
            1,),
        name = 'index_input')

    speed_input = Input(
        shape=(
            1,),
        name = 'speed_input')
            
   
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
        return_sequences = True,
        return_state = False,
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
        input_shape = (arch['n_units'][0],2),
        activation = arch['Dense_activation'],
        use_bias = True)(hidden_lstm_1)
    
    hidden_dense_2 = Dense(
        arch['n_units'][0],
        activation = arch['Dense_activation'],
        use_bias = True)(location_input)

    merge_layer_1 = concatenate([hidden_lstm_1, hidden_lstm_2])
    
    hidden_dense_2 = Dense(
        1,
        use_bias = True)(speed_input)
    
    hidden_dense_3 = Dense(
        1,
        use_bias = True)(index_input)

    hidden_dense_4 = Dense(
        arch['n_units'][1],
        activation = arch['Dense_activation'],
        use_bias = True)(merge_layer_1)

    merge_layer_2 = concatenate([hidden_dense_2, hidden_dense_3])

    merge_layer_3 = concatenate([merge_layer_1, merge_layer_2])

    hidden_dense_5 = Dense(
        arch['n_units'][2],
        activation = 'tanh',
        use_bias = True)(merge_layer_3)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_'+str(arch['sensors'][arch['target_sensor']]))(hidden_dense_5)
    model = Model(inputs = [peak_input, location_input, speed_input, index_input], outputs = output)
    return model
######################################################################################################
def set_up_model_test(arch):
    input1 = Input(shape=(1,),name='1')
    input2 = Input(shape=(1,),name='2')
    conc = concatenate([input1,input2])
    output1 = Dense(1,name='out')(conc)
    model = Model(inputs = [input1,input2], outputs = output1)
    return model 
