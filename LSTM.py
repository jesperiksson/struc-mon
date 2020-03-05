#Other files and classes
from util import *
from LSTMdata import * 
# Modules
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import metrics
from keras.utils import plot_model
from matplotlib import pyplot

class LongShortTermMemoryMachine():

    def __init__(self, batchStack, pred_sensor = 2, net_type = 'vanilla', feature_wise_normalization = False):

        """
        Args:

        """
        self.batchStack = batchStack
        # Feature-wise normalization of the data
        if feature_wise_normalization == True:
            pass
        else:
            pass
        self.n_batches = len(self.batchStack)
        self.pred_sensor = pred_sensor
        self.net_type = net_type
        self.n_sensors = 3#np.shape(self.batchStack['batch1'][0])
        self.activation='relu'
        self.loss='mse'
        if self.net_type == 'vanilla':
            model = Sequential()
            model.add(LSTM(100, activation = self.activation, return_sequences = True, input_shape=(None, self.n_sensors -1)))
            model.add(LSTM(100, activation = self.activation))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse', metrics=['mae','acc'])
            model.summary()
        #elif self.net_type == '': #And so forth

        self.model = model
        self.history = None
    

    def train(self, epochs = 200):
        
        """
        Args:
            samples : 1:k = n_samples
            nodes : 1:m = n_nodes
            accelerations : 1:n = n_accels
            [[[a111, ..., a11n], [a121, ..., a12n], ..., [a1m1, ..., a1mn]], ...,
             [[ak11, ..., ak1n], [ak21, ..., ak2n], ..., [akm1, ..., akmn]]]
        """
        def train_generator():
            train_data = np.array([])
            for i in range(len(self.batchStack)):
                key = 'batch'+str(i)
                data = self.batchStack[key].data
                if self.batchStack[key].category == 'train':
                    train_batch = np.array(self.batchStack[key].data)
                    targets = np.reshape(train_batch[self.pred_sensor,:], [np.shape(data)[1], 1])
                    patterns = np.reshape(np.delete(train_batch, self.pred_sensor, axis=0), [np.shape(data)[1], 1, 2])
                yield(patterns, targets)
                

        # fit model
        self.model.fit_generator(train_generator(), epochs, verbose=1)
#        self.history = self.model.fit(self.data_stack['trainset'].patterns, self.data_stack['trainset'].targets, epochs, verbose=0)

        return

    def predict(self):
        model_input = self.data_stack['testset'].patterns
        prediction = self.model.predict(model_input, batch_size = 40, verbose=0)
        print('\n Prediction:\n ',prediction)
        print('\n Real values:\n',self.data_stack['testset'].targets)
        pyplot.plot(self.history.history['mae'])
        #pyplot.show()
        return


