from util import *
import tensorflow as tf
'''Importing all necessary functions to create a LSTM Machine'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
#from keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences, metrics, initializers


class LongShortTermMemoryMachine():

    def __init__(self, trainset, testset, n_batches, net_type = 'vanilla'):

        """
        Args:
            net_type: which type of LSTM net (e.g. vanilla)
            time_window: Duration of sample
            n_accels: Number of accelerometers
            pred_accelerom: the accelerometer whose result is to be predicted
        """
        self.net_type = net_type
        self.n_timesteps = trainset.shape[1]
        self.n_batches = n_batches
        self.n_sensors = int(trainset.shape[0]/n_batches)
        self.activation='relu'
        self.loss='mse'
        if self.net_type == 'vanilla':
            model = Sequential()
            model.add(LSTM(self.n_timesteps, input_shape=(self.n_timesteps, self.n_sensors)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
        #elif self.net_type == '': #And so forth

        self.model = model
    

    def train(self, pattern, target, n_samples, epochs = 200):
        
        """
        Args: 
            trainset: Training dataset, containing data on this form:
            samples : 1:k = n_samples
            nodes : 1:m = n_nodes
            accelerations : 1:n = n_accels
            [[[a111, ..., a11n], [a121, ..., a12n], ..., [a1m1, ..., a1mn]], ...,
             [[ak11, ..., ak1n], [ak21, ..., ak2n], ..., [akm1, ..., akmn]]]
            lbl: the sample that is to be predicted (and therefore excluded from training)
            n_samples = number of samples contained in the trainset
        """

        # fit model
        model.fit(pattern, targets, epochs=epochs, verbose=0)

        return

    def predict(self, model_input):
        prediction = self.model.predict(model_input, verbose=0)
        print('\n Prediction:',prediction)
        return

