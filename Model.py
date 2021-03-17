# Standard packages
import time
import importlib as il
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# External packages
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import tensorflow_io as tfio



# Self made modules
import config
from WindowGenerator import *
from ReportGenerator import ReportGenerator

tf.get_logger().setLevel('ERROR')

class Model(): # Methods and features shared across all predictive models 
    def __init__(self,settings):
        
        self.settings = settings
        self.name = settings.name
        self.report_generator = ReportGenerator(settings)
      
    def train_test_split(self,df): # makes a dataframe out of all the smaller dataframes
        
        n = len(df)
        self.train_df = df[0:int(n*self.data_split.train)]
        self.val_df = df[
            int(n*self.data_split.train):int(n*self.data_split.validation) + int(n*self.data_split.train)
            ]
        self.test_df = df[-int(n*self.data_split.train):]

        
    def detect_outliers(self): # Calculates p-value for residual
        statistic , pvalue = stats.normaltest(self.residual)
        np.set_printoptions(precision=4)
        
    def plot_outliers(self): # Plots histogram of predictions an a scatterplot
        #fig, ax  = plt.subplots(nrows=1,ncols=1)
        plt.hist(
            self.mae.numpy()[:,0,0],
            bins = 100)
        plt.title('Prediction errors')
        plt.ylabel(f'Prediction error for {self.settings_nn.plot_target}')
        plt.show()
        
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
        
class StatModel(Model): # Purely statistical model to be used as baseline
    def __init__(self,settings):
        super().__init__(settings)
        
    def setup_model(self):
        sys.path.append(config.preset_path)
        module = il.import_module(self.settings.preset) # Import the specified model from file with same name
        learned = None # TBI            
        

class NeuralNet(Model): # Methods and features shared among all Keras Neural Nets
    def __init__(self,settings):
        super().__init__(settings)

    def setup_nn(self, plot_model=False):
        sys.path.append(config.preset_path)
        module = il.import_module(self.settings.preset) # Import the specified model from file with same name
        learned = None # TBI       
          
        # Record settings from the neural net module    
        self.settings_nn = module.Settings_nn()
        self.settings_train = module.Settings_train()
        self.settings_eval = module.Settings_eval()
        self.settings_test = module.Settings_test()
        self.data_split = module.DataSplit()
        self.nn = module.set_up_model(self.settings_nn)     
        # learning rate decay
        # learning algortihm
        # other stuff
        self.nn.compile( # compile the net
            loss = self.settings_train.loss,
            optimizer = self.settings_train.optimizer,
            metrics = self.settings_train.metrics)      
        if plot_model:
            tf.keras.utils.plot_model(
                self.nn, 
                config.saved_path+self.settings.name+'/model_plot.jpeg',
                show_shapes = True,
                show_layer_names = True,
                dpi = 150)
        return learned
            
    def train(self): # Train the neural net
        for example_inputs, example_labels in self.time_series.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')
        tic = time.time()  
        self.history = self.nn.fit(
            self.time_series.train,
            epochs = self.settings_train.epochs,
            batch_size = self.settings_train.batch_size,
            verbose = self.settings_nn.verbose)
        self.toc = time.time() - tic
        self.training_report = self.report_generator.generate_training_report(self)

    def evaluate(self): # Evaluate the neural net
        self.test_loss = self.nn.evaluate(
            self.time_series.val,
            batch_size = self.settings_eval.batch_size,
            verbose = self.settings_nn.verbose
        )
        
    def test(self): # Test the neural net
        prediction = self.nn.predict(
            self.time_series.test,
            batch_size = self.settings_test.batch_size,
            verbose = self.settings_nn.verbose
        )
        ground_truth = tf.concat([y for x, y in self.time_series.test], axis=0)
        #self.residual = tf.math.subtract(tf.squeeze(prediction),tf.squeeze(ground_truth),name='residual')
        self.mae = tf.math.subtract(prediction,ground_truth,name='residual')
        
        
    def plot_example(self): # Plot an input-output example
        self.time_series.plot(
            plot_col = self.settings_nn.plot_target,
            model = self.nn)
        plt.show()
        
            
    def plot_history(self): # Plot the training history for each metric
        key_list = list(self.history.history.keys())
        [plt.plot(self.history.history[key]) for key in key_list]
        plt.legend(key_list)
        plt.title(f'Training history for {self.name}, trained for {self.settings_train.epochs} epochs. Elaspsed time: {self.toc}')
        plt.xlabel('epoch')
        plt.ylabel('error') 
        plt.savefig(config.saved_path+self.settings.name+''.join(self.settings.sensors))
        plt.show()
        
           
    def save_nn(self,overwrite=False):
        # Backup???
        path = config.saved_path+self.settings.name
        self.nn.save(
            filepath = path,
            overwrite = overwrite,
            include_optimizer = True,
            save_format = 'tf')
            
        f = open(config.saved_path+self.settings.name+'/report.txt','a')
        f.write(self.training_report)
        f.close()
            
            
    def load_nn(self):
        if self.settings.name not in os.listdir(config.saved_path):
            print(f'No saved model named {self.settings.name}')
            #raise Exception('No module to load')
        else:
            loaded_nn = tf.keras.models.load_model(
                filepath = config.saved_path+self.settings.name,
                compile = True)
            self.nn = loaded_nn
            print(f"Loaded {self.nn}")

        
class TimeSeriesNeuralNet(NeuralNet): # For RNNs, CNNs, etc. Obsolete?
    def __init__(self,settings):
        super().__init__(settings)
     

     
    

