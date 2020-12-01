
# External packages
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Standard packages
import time
import importlib as il
import sys
import os

# Self made modules
import config
import Make_model
from WindowGenerator import *


class Model(): # Methods and features shared across all predictive models 
    def __init__(self,settings,existing_model):
        
        self.settings = settings
        self.name = settings.name
        self.existing_model = existing_model  
      
    def make_dataframe(self,series_stack): # makes a dataframe out of all the smaller dataframes
        # concatenate dataframes into a big dataframe
        try: # See if the series_stack is populated
            big_df = pd.DataFrame(columns=series_stack.stack[0].data.columns)
        except IndexError:
            print('There is no data to read')
            return
        for i in range(len(series_stack.stack)):
            small_df = series_stack.stack[i].data
            big_df = big_df.append(small_df)
        #self.dataframe = big_df[self.settings.features]
        n = len(big_df)
        self.train_df = big_df[0:int(n*self.data_split.train)]
        self.val_df = big_df[
            int(n*self.data_split.train):int(n*self.data_split.validation) + int(n*self.data_split.train)
            ]
        self.test_df = big_df[-int(n*self.data_split.train):]
        
    def show_plot(self): # Shows the latest plot
        plt.show()
        
        
        
        # methods for visualizing data

class NeuralNet(Model): # Methods and features shared among all Keras Neural Nets
    def __init__(self,settings,existing_model):
        super().__init__(settings,existing_model)

    def setup_nn(self, print_summary=False):
        sys.path.append(config.preset_path)
        if self.settings.use_preset == True:          
            module = il.import_module(self.settings.preset) # Import the specified model from file with same name
            learned = None # TBI
        elif self.settings.use_preset == False:
            module = il.import_module(self.settings.template) 
            learned = None # TBI 
        else: 
            raise Exception('use_preset ambiguos') # should never happen           
            
        self.settings_nn = module.Settings_nn()
        self.settings_train = module.Settings_train()
        self.settings_eval = module.Settings_eval()
        self.settings_test = module.Settings_test()
        self.data_split = module.DataSplit()
        self.nn = module.set_up_model(self.settings_nn)     
        # learning rate decay
        # learning algortihm
        # other stuff
        self.nn.compile(
            loss = self.settings_train.loss,
            optimizer = self.settings_train.optimizer,
            metrics = self.settings_train.metrics)      
        if print_summary:
            self.nn.summary() 
        return learned
            
    def train(self):
        for example_inputs, example_labels in self.time_series.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')
        self.history = self.nn.fit(
            self.time_series.train,
            epochs = self.settings_train.epochs,
            batch_size = self.settings_train.batch_size,
            verbose = self.settings_train.verbose)

    def evaluate(self):
        self.test_loss = self.nn.evaluate(
            self.time_series.val,
            batch_size = self.settings_eval.batch_size,
            verbose = self.settings_eval.verbose
        )
        
    def predict(self):
        self.performance = self.nn.evaluate(
            self.time_series.test,
            batch_size = self.settings_test.batch_size,
            verbose = self.settings_test.verbose
        )
        
    def predict_single_sample(self,n_samples = 1):
        random_sample = self.time_series.test.take(n_samples)
        prediction = self.nn.predict(random_sample)
        print(random_sample,prediction)
        
    def plot_example(self):
        self.time_series.plot(
            plot_col = self.settings_nn.plot_target,
            model = self.nn)
        plt.show()
        
            
    def plot_history(self):
        key_list = list(self.history.history.keys())
        [plt.plot(self.history.history[key]) for key in key_list]
        plt.legend(key_list)
        plt.title('Training history for '+self.name+', trained for '+self.settings_train.epochs+'epochs')
        plt.xlabel('epoch')
        plt.ylabel('error') 
        plt.show()
        
    def plot_test_loss(self):
        print(self.test_loss)
           
    def save_nn(self,overwrite=False):
        #print(config.saved_path+self.settings.name)
        if self.settings.name not in os.listdir(config.saved_path):
            os.mkdir(config.saved_path+self.settings.name)
        else:
            pass # TODO: prompt for over writing
        self.nn.save(
            filepath = config.saved_path+self.settings.name,
            overwrite = overwrite,
            include_optimizer = True,
            save_format = 'tf')
            
    def load_nn(self):
        if self.settings.name not in os.listdir(config.saved_path):
            raise Exception('No module to load')
        else:
            loaded_nn = tf.keras.models.load_model(
                filepath = config.saved_path+self.settings.name,
                compile = True)
            self.nn = loaded_nn
        
class GenericNeuralNet(NeuralNet): # Manual settings (not 'genetic')
    def __init__(self,settings,existing_model): # OBSOLETE
        super().__init__()   
        
    def make_data_set(self): # An tf.data.Dataset object made from the dataframe
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(self.dataframe[self.settings['features']].values, dtype=tf.float32),
                tf.cast(self.dataframe[self.settings['target']].values, dtype=tf.float32)
                )
            )
    
    def make_iterator(self):
        pass
        
    def make_model(self):
        pass

        
class TimeSeriesNeuralNet(NeuralNet): # For RNNs, CNNs, etc.
    def __init__(self,settings,existing_model):
        super().__init__(settings,existing_model)
        
    def make_timeseries_dataset(self, print_shape=False):
        self.time_series = WindowGenerator(
            input_width = self.settings_nn.input_time_steps,
            label_width = self.settings_nn.target_time_steps,
            shift = self.settings_nn.shift,
            train_df = self.train_df[self.settings_nn.features],
            val_df = self.val_df[self.settings_nn.features],
            test_df = self.test_df[self.settings_nn.features],
            label_columns = self.settings_nn.features,
            train_batch_size = self.settings_train.batch_size,
            eval_batch_size = self.settings_eval.batch_size,
            test_batch_size = self.settings_test.batch_size)
        if print_shape:
            for example_inputs, example_labels in self.time_series.train.take(1):
                print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
                print(f'Labels shape (batch, time, features): {example_labels.shape}')
     
def rmse(true, prediction):
    return backend.sqrt(backend.mean(backend.square(prediction - true), axis=-1))
def rmse_np(true, prediction):
    return np.sqrt(np.mean(np.square(prediction - true), axis=-1))
    

