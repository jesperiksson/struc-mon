#Other files and classes
from util import *
from MLPbatch import * 
# Modules
import tensorflow as tf
import keras
import os
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, concatenate, Activation 
from keras import metrics, callbacks, regularizers
from keras.utils import plot_model
from matplotlib import pyplot
from keras.optimizers import RMSprop

class NeuralNet():
    def __init__(self,
                 architecture,
                 data_split,
                 name,
                 pred_sensor = 2,
                 n_sensors = 3,
                 early_stopping = True,
                 existing_model = False):

        self.architecture = architecture
        self.ad_hoc_batchStack = None
        self.data_split = data_split
        self.name = name
        self.model_type = self.architecture['model_type']
        self.activation = self.architecture['MLPactivation']
        if early_stopping == True:
            '''
            https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
            '''
            self.early_stopping = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0.00000000001, 
                                                 patience=2,
                                                 verbose=1,
                                                 mode='auto')]
        else:
            self.early_stopping = None
        self.loss='mse'
        self.existing_model = existing_model           
        if self.existing_model == False:
            acceleration_input = Input(shape=(architecture['n_pattern_steps'], ), name='acceleration_input')
            accel_layer = Dense(self.architecture['n_units'][0],
                                activation = self.architecture['MLPactivation'],
                                use_bias = True,
                                kernel_regularizer = regularizers.l2(0.0001))(acceleration_input)
            element_input = Input(shape(1), name='element_input') # damaged element
            speed_input = Input(shape(1), name='speed_input') 
            damage_input = Input(shape=(1), name='damage_input') # Ratio of the Young's modulous(210GPa)
            x = concatenate([accel_layer, element_input, speed_input, damage_input])
            x = Dense(architecture['n_units'][1], activation='sigmoid')(x)
            if architecture['target'] == 'E':
                output = Dense(1, activation='sigmoid', name='E_output')(x)
            elif architecture['target'] == 'acceleration':
                output = Dense(architecture['n_target_steps'], activation='sigmoid', name='acceleration_output')
            self.model = Model(inputs=(acceleration_input, element_input, speed_input, damage_input)
                               outputs = output)
            model.compile(optimizer='adam', loss='mse', metrics=['mse','acc'])
            model.summary()             
            self.model = model

        elif self.existing_model == True:
            model_path = 'models/'+self.name+'.json'
            weights_path = 'models/'+self.name+'.h5'
            json_file = open(model_path)
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(weights_path)
            self.model = loaded_model
            print('\n Loaded model: ', name)
            self.model.compile(optimizer='adam')
            self.model.summary()
        else:
            raise Error
        self.history = None
        self.score = None

    def train(self, epochs = 200):
        '''
        Reshapes the data to the form 
        0 [x_00 = a_1, x_01 = a_1+delta, ..., x_0(n_pattern_steps) = a_(delta*n_pattern_steps)]
        1 [x_10 = a_2, x_11 = a_2+delta, ..., x_1(n_pattern_steps) = a_(delta*n_pattern_steps)]
        .
        .
        n_series  [x_n_series0 = a_n_series...]
        '''
        delta = self.architecture['delta']
        n_pattern_steps = self.architecture['n_pattern_steps']
        n_target_steps = self.architecture['n_target_steps']
        n_series = int(self.batchStack['batch1'].diff)-int(delta*n_pattern_steps)
 
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        def generator(self, task):    
            i = 0
            while True:
                key = 'batch'+str(i%len(self.batchStack))
                i+=1
                if self.batchStack[key].category == task:
                    for j in range(n_series):                
                        pattern_indices = np.arange(j,j+(delta)*n_pattern_steps,delta)
                        target_indices = j+delta*n_pattern_steps
                        patterns[j,:] = self.batchStack[key].batch[self.architecture['sensor_key']][pattern_indices]
                        targets[j,:] = self.batchStack[key].batch[self.architecture['sensor_key']][target_indices]
                    yield(patterns, targets)
                else:
                    pass
        self.history = self.model.fit_generator( generator(self,'train'), 
                                                 steps_per_epoch=1, 
                                                 epochs=epochs, 
                                                 verbose=1,
                                                 callbacks=self.early_stopping, 
                                                 validation_data = generator(self,'validation'),
                                                 validation_steps=10)
        self.model.summary()
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
        self.used_epochs = len(self.val_loss)
        return

    def train_ad_hoc(self, batchStack, epochs = 10):
        delta = self.architecture['delta']
        n_pattern_steps = self.architecture['n_pattern_steps']
        n_target_steps = self.architecture['n_target_steps']
        
 
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])          
        
        for i in range(self.architecture['speeds']):
            key = 'batch'+str(i%len(batchStack))
            batch = batchStack['key']
            n_series = int(np.shape(batch.data)[1])-int(delta*n_pattern_steps)
            for j in range(n_series):                
                pattern_indices = np.arange(j,j+(delta)*n_pattern_steps,delta)
                target_indices = j+delta*n_pattern_steps
                patterns[j,:] = batchStack[key].batch[self.architecture['sensor_key']][pattern_indices]
                targets[j,:] = batchStack[key].batch[self.architecture['sensor_key']][target_indices]

            self.history = model.train_on_batch({  'acceleration_input' : patterns,
                                                    'element_input' : batch.element,
                                                    'speed_input': batch.speed,
                                                    'damage_input' : batch.damage_state}
                                                    {self.architecture['target'] : targets},
                                                    epochs,
                                                    verbose = 1,
                                                    validation_split = self.architecture['data_split'][1])

        def generator(self, task):    
            i = 0
            while True:
                key = 'batch'+str(i%len(self.batchStack))
                i+=1
                if self.batchStack[key].category == task:
                    for j in range(n_series):                
                        pattern_indices = np.arange(j,j+(delta)*n_pattern_steps,delta)
                        target_indices = j+delta*n_pattern_steps
                        patterns[j,:] = self.batchStack[key].batch[self.architecture['sensor_key']][pattern_indices]
                        targets[j,:] = self.batchStack[key].batch[self.architecture['sensor_key']][target_indices]
                    yield(patterns, targets)
                else:
                    pass
        self.history = self.model.fit_generator( generator(self,'train'), 
                                                 steps_per_epoch=1, 
                                                 epochs=epochs, 
                                                 verbose=1,
                                                 callbacks=self.early_stopping, 
                                                 validation_data = generator(self,'validation'),
                                                 validation_steps=10)
        self.model.summary()
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
        self.used_epochs = len(self.val_loss)
        return
    
    def evaluation(self):
        delta = self.architecture['delta']
        n_pattern_steps = self.architecture['n_pattern_steps']
        n_target_steps = self.architecture['n_target_steps']
        n_series = int(self.batchStack['batch1'].diff)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        def generator():
            i = 0
            while True:
                key = 'batch'+str(i%len(self.batchStack))
                i += 1
                if self.batchStack[key].category == 'test':
                    for j in range(n_series):
                        pattern_indices = np.arange(j,j+(delta)*n_pattern_steps,delta)
                        target_indices = j+delta*n_pattern_steps
                        patterns[j,:] = self.batchStack[key].batch[self.architecture['sensor_key']][pattern_indices]
                        targets[j,:] = self.batchStack[key].batch[self.architecture['sensor_key']][target_indices]
                    yield(patterns, targets)
                else:
                    pass
        self.score = self.model.evaluate_generator(generator(),
                                                     steps = 1,
                                                     verbose = 1)
        print('Model score: ',self.score)
        return

    def prediction(self, batch, external_batch = False, batchStack = None):
        if external_batch == False:
            batchStack = self.batchStack
        delta = self.architecture['delta']
        n_pattern_steps = self.architecture['n_pattern_steps']
        n_target_steps = self.architecture['n_target_steps']
        n_series = int(batchStack['batch1'].diff)-int(delta*n_pattern_steps)
        key = 'batch'+str(batch%len(batchStack))
        n_series = int(batchStack[key].diff)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        if batchStack[key].category == 'test':
            for i in range(n_series):
                pattern_indices = np.arange(i,i+(delta)*n_pattern_steps,delta)
                target_indices = i+delta*n_pattern_steps
                patterns[i,:] = self.batchStack[key].batch[self.architecture['sensor_key']][pattern_indices]
                targets[i,:] = self.batchStack[key].batch[self.architecture['sensor_key']][target_indices]
            prediction = self.model.predict(patterns, batch_size=10, verbose=1)
            plot_prediction(self, prediction, batch, targets)
        else:
            for i in range(n_series):
                pattern_indices = np.arange(i,i+(delta)*n_pattern_steps,delta)
                target_indices = i+delta*n_pattern_steps
                patterns[i,:] = batchStack[key].batch[self.architecture['sensor_key']][pattern_indices]
                targets[i,:] = batchStack[key].batch[self.architecture['sensor_key']][target_indices]
            return self.model.predict(patterns, batch_size=10, verbose=0), targets            
        
        return

    def get_H_score(self, mse_threshold):
        right = 0
        wrong = 0
        delta = self.architecture['delta']
        n_pattern_steps = self.architecture['n_pattern_steps']
        n_target_steps = self.architecture['n_target_steps']
        n_series = int(self.batchStack['batch1'].diff)-int(delta*n_pattern_steps)
        patterns = np.empty([n_series,n_pattern_steps])
        targets = np.empty([n_series,n_target_steps])
        for i in range(len(self.batchStack)):
            key = 'batch'+str(i%len(self.batchStack))
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
