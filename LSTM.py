#Other files and classes
from util import *
from LSTMdata import * 
# Modules
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
#from keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences, metrics, initializers


class LongShortTermMemoryMachine():

    def __init__(self, data, train_percentage, split_mode, pred_sensor, n_batches , net_type = 'vanilla'):

        """
        Args:

        """
        self.data = data
        self.n_batches = n_batches
        self.net_type = net_type
        self.train_percentage = train_percentage
        self.split_mode = split_mode

        self.data_stack = {
            'trainset' : LSTMdata(data ,'train' ,n_batches ,train_percentage ,split_mode),
            'testset' : LSTMdata(data ,'test' ,n_batches ,train_percentage ,split_mode)
        }
        self.n_timesteps = self.data.shape[1]
        self.n_sensors = int(self.data.shape[0]/self.n_batches)
        self.activation='relu'
        self.loss='mse'
        if self.net_type == 'vanilla':
            model = Sequential()
            model.add(LSTM(self.n_timesteps, input_shape=(self.n_sensors,self.n_timesteps-1)))
            model.add(Dense(self.n_sensors))
            model.compile(optimizer='adam', loss='mse')
        #elif self.net_type == '': #And so forth

        self.model = model
    

    def train(self, epochs = 200):
        
        """
        Args:
            samples : 1:k = n_samples
            nodes : 1:m = n_nodes
            accelerations : 1:n = n_accels
            [[[a111, ..., a11n], [a121, ..., a12n], ..., [a1m1, ..., a1mn]], ...,
             [[ak11, ..., ak1n], [ak21, ..., ak2n], ..., [akm1, ..., akmn]]]
        """

        # fit model
        self.model.fit(self.data_stack['trainset'].patterns, self.data_stack['trainset'].targets, epochs, verbose=0)

        return

    def predict(self):
        model_input = self.data_stack['testset'].patterns
        prediction = self.model.predict(model_input, batch_size = 40, verbose=0)
        print('\n Prediction:\n',prediction)
        print('\n Real values:\n',self.data_stack['testset'].targets)
        return

