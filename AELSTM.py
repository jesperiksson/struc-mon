#Other files and classes
from util import *
from Databatch import * 
# Modules
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
                 early_stopping = True,
                 existing_model = False,
                 sensor_to_predict = 0):

        self.arch = arch
        self.name = name
        self.target_sensor = self.arch['sensors'][self.arch['target_sensor']]
        #self.pattern_sensors = [None]*len([self.arch['pattern_sensors']])
        self.pattern_sensors = np.arange(0,len(self.arch['pattern_sensors']),1)

        if early_stopping == True:
            self.early_stopping = [keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=0, 
                patience=1,
                verbose=1,
                mode='auto',
                restore_best_weights=True)]
        else:
            self.early_stopping = None
        if arch['learning_rate_schedule'] == True:
            self.learning_rate_sheduler = LearningRateScheduler(step_decay)
            def step_decay(epoch):
                initial_rate = arch['learning_rate']
                drop = 0.5
                epochs_drop = 10
                learning_rate = initial_rate * np.power(drop, np.floor((1+epoch)/epochs_drop))
                return learning_rate

        else:
            self.learning_rate_scheduler = None
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
            if series.category == 'train' or series.category == 'validation':
                X, Y = generator(self, series)
                patterns = {'speed_input' : np.array([series.normalized_speed])}
                for j in range(len(self.arch['active_sensors'])):
                    patterns.update({
                        'accel_input_'+str(self.arch['pattern_sensors'][j]) : X
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
        self.model.summary()
        #self.used_epochs = len(self.loss)   
        return

    def evaluation(self, series_stack):
        for i in range(len(series_stack)):
            series = series_stack[self.arch['preprocess_type']]['batch'+str(i%len(series_stack))]
            if series.category == 'test':
                X, Y = generator(self, series)
                self.score = self.model.evaluate(
                    x = X,
                    y = Y,
                    batch_size = self.arch['batch_size'],
                    verbose = 1,
                    return_dict = True)
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
                speeds.extend([series.speed])
                scores.extend([score])
        results = {
            'scores' : scores[1:],
            'speeds' : speeds[1:],
            'steps' : self.arch['n_target_steps']
        }

        return results

    def prediction(self, manual):
        series = manual['stack'][self.arch['preprocess_type']]['batch'+str(manual['series_to_predict']%len(manual['stack']))]
        X, Y = generator(self, series)
        predictions = self.model.predict(
            X,
            batch_size = self.arch['batch_size'],
            verbose = 1,
            steps = steps)

        prediction = {
            'prediction' : predictions,
            'hindsight' : Y,
            'steps' : steps,
        }
        
        return prediction

    def modify_model(self):
        self.used_epochs = len(self.loss)  


def generator(self, batch):
    '''
    Generator for when to use Peak accelerations and locations
    Each series of data is so big it has to be broken down
    '''
    steps = get_steps(self, batch)
    X = np.empty([steps,self.arch['n_pattern_steps'], len(self.arch['pattern_sensors'])])
    Y = np.empty([steps,self.arch['n_pattern_steps']])
    for j in range(len(self.arch['pattern_sensors'])):     
        for k in range(steps):    
            start = k*self.arch['pattern_delta']
            finish = k*self.arch['pattern_delta']+self.arch['delta']*self.arch['n_pattern_steps']
            X[k,:,j] = batch.data[j][start:finish]
            Y[k,:] = batch.data[j][1+start:1+finish]
    return X, Y
    '''
    Generates inputs with shape [n_batches, n_timesteps, features]
    When there is not enough samples left to form a batch, the last batches will not be incorporated.
    TBD: in order to use several sensors, the sensor location needs to be included in the input
    '''



def get_targets(self, peak_target):
    targets = {
        'accel_output_'+str(self.target_sensor) : 
        np.reshape(peak_target,[1, self.arch['n_target_steps']])
    }

def get_steps(self, series):
    steps = int(
        np.floor(
            (series.steps[self.target_sensor]-self.arch['n_pattern_steps'])/self.arch['pattern_delta']
        )
    )
    return steps    

def rmse(true, prediction):
    return backend.sqrt(backend.mean(backend.square(prediction - true), axis=-1))


#######################################################################################################
def set_up_model5(arch):
    '''
    Peaks and their positions deltas as inputs
    '''
    peak_input = Input(
        shape=(
            arch['n_pattern_steps'], 
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
            arch['n_pattern_steps'],
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
        arch['n_target_steps'], 
        activation='tanh',
        return_sequences = False, 
        name='peak_output_'+str(arch['sensors'][arch['target_sensor']]))(decoded_1)

    output = Reshape(
        input_shape = (0,arch['n_pattern_steps'],1),
        target_shape = (arch['n_pattern_steps'],1))(hidden)

    model = Model(inputs = peak_input, outputs = output)
    return model
######################################################################################################

