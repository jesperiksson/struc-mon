#Other files and classes
from util import *
from LSTMdata import * 
# Modules
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import metrics, callbacks
from keras.utils import plot_model
from matplotlib import pyplot
from keras.optimizers import RMSprop

class LongShortTermMemoryMachine():

    def __init__(self,
                 architecture,
                 batchStack,
                 data_split,
                 pred_sensor = 2,
                 n_sensors = 3,
                 feature_wise_normalization = False,
                 early_stopping = True):

        """
        Args:

        """
        self.architecture = architecture
        self.batchStack = batchStack
        self.data_split = data_split
        # TODO Feature-wise normalization of the data
        if feature_wise_normalization == True:
            pass
        else:
            pass
        if early_stopping == True:
            '''
            https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
            '''
            self.early_stopping = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 min_delta=0, 
                                                 patience=0,
                                                 verbose=1,
                                                 mode='auto')]
        else:
            self.early_stopping = None
        self.n_batches = len(self.batchStack)
        self.pred_sensor = pred_sensor
        self.n_sensors = n_sensors
        self.activation='relu'
        self.loss='mse'           
 
        model = Sequential()
        if self.architecture['direction'] == 'uni':            
            model.add(LSTM(self.architecture['n_units'][0], 
                                             activation = self.activation,
                                             return_sequences = True,
                                             input_shape=(None, self.n_sensors -1)))
            for i in range(self.architecture['n_LSTM_layers']-1):
                model.add(LSTM(self.architecture['n_units'][i+1]))

        elif self.architecture['direction'] == 'uni':
            model.add(Bidirectional(LSTM(self.architecture['n_units'][0], 
                                             activation = self.activation,
                                             return_sequences = True,
                                             input_shape=(None, self.n_sensors -1))))
            for i in range(self.architecture[n_LSTM_layers]-1):
                model.add(Bidirectional(LSTM(self.architecture['n_units'][i+1])))

        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae','acc'])
        model.summary()            
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
        self.epochs = epochs
        def train_generator():
            train_data = np.array([])
            i = 0
            while True:
                i += 1 
                key = 'batch'+str(i%len(self.batchStack))
                data = self.batchStack[key].data
                if self.batchStack[key].category == 'train':
                    train_batch = np.array(self.batchStack[key].data)
                    targets = np.reshape(train_batch[self.pred_sensor,:],
                                         [np.shape(data)[1], 1])
                    patterns = np.reshape(np.delete(train_batch,
                                          self.pred_sensor, axis=0),
                                          [np.shape(data)[1], 1, 2])
                    yield(patterns, targets)
                
        # TODO validation_generation
        def validation_generator():
            validation_data = np.array([])
            i = 0
            while True:
                i +=1
                key = 'batch'+str(i%len(self.batchStack))
                data = self.batchStack[key].data
                if self.batchStack[key].category == 'validation':
                    validation_batch = np.array(self.batchStack[key].data)
                    targets = np.reshape(validation_batch[self.pred_sensor,:],
                                         [np.shape(data)[1], 1])
                    patterns = np.reshape(np.delete(validation_batch,
                                          self.pred_sensor, axis=0),
                                          [np.shape(data)[1], 1, 2])
                    yield(patterns, targets)


        # fit model
        self.history = self.model.fit_generator(train_generator(),
                                                steps_per_epoch = int(self.n_batches*self.data_split[0]/100),
                                                epochs = self.epochs,
                                                verbose=1,
                                                callbacks = self.early_stopping, 
                                                validation_data = validation_generator(),
                                                validation_steps = int(self.n_batches*self.data_split[1]/100)
                                                )
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
        
        return

    def evaluate(self):
        '''
        Args:
        '''
        def evaluation_generator():
            test_data = np.array([]);
            i = 0
            while True:
                i += 1
                key = 'batch'+str(i%len(self.batchStack))
                data = self.batchStack[key].data
                if self.batchStack[key].category == 'test':
                    test_batch = np.array(self.batchStack[key].data)
                    targets = np.reshape(test_batch[self.pred_sensor,:],
                                         [np.shape(data)[1], 1])
                    patterns = np.reshape(np.delete(test_batch,
                                          self.pred_sensor, axis=0),
                                          [np.shape(data)[1], 1, 2])                    
                    yield (patterns, targets)
        evaluation = self.model.evaluate_generator(evaluation_generator(), 
                                                   steps = int(self.n_batches*self.data_split[2]/100), 
                                                   verbose = 1
                                                   )
        return evaluation

    def plot_loss(self):
        plt.figure()
        plt.plot(range(1,self.epochs+1), self.loss, 'bo', label='Training loss')
        plt.plot(range(1,self.epochs+1), self.val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
