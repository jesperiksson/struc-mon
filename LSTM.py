# Other files and classes
from util import *
from LSTMbatch import * 
# Modules
import tensorflow as tf
import keras
import os
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, BatchNormalization, Bidirectional
from keras import metrics, callbacks
from keras.utils import plot_model
from matplotlib import pyplot
from keras.optimizers import RMSprop

class NeuralNet():
    def __init__(self,
                 architecture,
                 batchStack,
                 data_split,
                 name,
                 pred_sensor = 2,
                 n_sensors = 3,
                 feature_wise_normalization = False,
                 early_stopping = True,
                 existing_model = False):

        """
        Args:

        """
        self.architecture = architecture
        self.batchStack = batchStack
        self.data_split = data_split
        self.name = name
        self.model_type = self.architecture['model_type']
        if early_stopping == True:
            '''
            https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
            '''
            self.early_stopping = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0.0001, 
                                                 patience=2,
                                                 verbose=1,
                                                 mode='auto')]
        else:
            self.early_stopping = None
        self.n_batches = len(self.batchStack)
        self.pred_sensor = pred_sensor
        self.n_sensors = n_sensors
        if feature_wise_normalization == True:
            for i in range(len(self.batchStack)):
                for j in range(self.n_sensors):
                    data = self.batchStack['batch'+str(i)].data[j]
                    mean = data.mean(axis=0)
                    data -= mean
                    std = data.std(axis=0)
                    data /= std
                    self.batchStack['batch'+str(i)].data[j] = data
        else:
            pass
        self.activation='relu'
        self.loss='mse'
        self.existing_model = existing_model           
        if self.existing_model == False: 
            if self.architecture['direction'] == 'uni':            
                model.add(LSTM(self.architecture['n_units'][0], 
                               activation = self.activation,
                               return_sequences = True,
                               input_shape=(None, n_features)))
                for i in range(self.architecture['n_LSTM_layers']-1):
                    model.add(LSTM(self.architecture['n_units'][i+1]))

            elif self.architecture['direction'] == 'bi':
                model.add(Bidirectional(LSTM(self.architecture['n_units'][0], 
                                                 activation = self.activation,
                                                 return_sequences = True,
                                                 input_shape=(None, n_features))))
                for i in range(self.architecture['n_LSTM_layers']-1):
                    model.add(Bidirectional(LSTM(self.architecture['n_units'][i+1])))

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
            self.model.summary()
        else:
            raise Error

    def train(self, epochs = 200): # Training and validating a model on the respective datasets
        
        """
        Args:
            samples : 1:k = n_samples
            nodes : 1:m = n_nodes
            accelerations : 1:n = n_accels
            [[[a111, ..., a11n], [a121, ..., a12n], ..., [a1m1, ..., a1mn]], ...,
             [[ak11, ..., ak1n], [ak21, ..., ak2n], ..., [akm1, ..., akmn]]]
        """
        self.epochs = epochs
        def train_generator(self):
            i = 0
            task = 'train'
            while True:
                i += 1 
                key = 'batch'+str(i%len(self.batchStack))
                data = self.batchStack[key].data
                yield generator(self, data, key, task)
                
        def validation_generator(self):
            i = 0
            task = 'validation'
            while True:
                i +=1
                key = 'batch'+str(i%len(self.batchStack))
                data = self.batchStack[key].data
                yield generator(self, data, key, task)

        def generator(self, data, key, task):
            if self.batchStack[key].category == task:
                if self.architecture['prediction'] == 'entire_series':
                    validation_batch = np.array(self.batchStack[key].data)
                    targets = np.reshape(validation_batch[self.pred_sensor,:],
                                         [1, np.shape(data)[1], 1])
                    patterns = np.reshape(np.delete(validation_batch,
                                          self.pred_sensor, axis=0),
                                          [1, np.shape(data)[1], 2])
                elif self.architecture['prediction'] == 'end_of_series':
                    train_batch = np.array(self.batchStack[key].data[self.pred_sensor])
                    split = -self.architecture['n_pred_units']
                    targets = np.array(train_batch[split-1:-1])
                    targets = np.reshape(targets, [1, np.shape(targets)[0], 1])
                    patterns = np.array(train_batch[:split])  
                    patterns = np.reshape(patterns, [1, np.shape(patterns)[0], 1])  
                return(patterns, targets)

        # fit model
        self.history = self.model.fit_generator(train_generator(self),
                                                steps_per_epoch = int(self.n_batches*self.data_split[0]/100),
                                                epochs = self.epochs,
                                                verbose=1,
                                                callbacks = self.early_stopping, 
                                                validation_data = validation_generator(self),
                                                validation_steps = int(self.n_batches*self.data_split[1]/100))
        self.model.summary()
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
        self.used_epochs = len(self.val_loss)
        
        return
