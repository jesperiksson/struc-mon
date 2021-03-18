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
#import tensorflow_probability as tfp



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

        
    def detect_outliers(self): # Calculates p-value for residual
        statistic , pvalue = stats.normaltest(self.residual)
        np.set_printoptions(precision=4)
        
    def plot_outliers(self): # Plots histogram of predictions an a scatterplot
        #fig, ax  = plt.subplots(nrows=1,ncols=1)
        plt.hist(
            self.residual.numpy()[:,0,0],
            bins = 100)
        plt.title('Prediction errors')
        plt.ylabel(f'Prediction error for {self.settings_model.plot_target}')
        plt.show()
        
    def make_timeseries_dataset(self, data, print_shape=False):
        self.time_series = WindowGenerator(
            input_width = self.settings_model.input_time_steps,
            label_width = self.settings_model.target_time_steps,
            shift = self.settings_model.shift,
            train_df = data.train_df[self.settings_model.features],
            val_df = data.val_df[self.settings_model.features],
            test_df = data.test_df[self.settings_model.features],
            label_columns = self.settings_model.targets,
            train_batch_size = self.settings_train.batch_size,
            eval_batch_size = self.settings_eval.batch_size,
            test_batch_size = self.settings_test.batch_size)
        try:
            if print_shape:
                for example_inputs, example_labels in self.time_series.train.take(1):
                    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
                    print(f'Labels shape (batch, time, features): {example_labels.shape}')
        except ValueError:
            for example_inputs, example_labels in self.time_series.train.test(1):
                print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
                print(f'Labels shape (batch, time, features): {example_labels.shape}')
        
class StatModel(Model): # Purely statistical model to be used as baseline
    def __init__(self,settings):
        super().__init__(settings)
        
    def setup(self):
        sys.path.append(config.preset_path)
        module = il.import_module(self.settings.preset) # Import the specified model from file with same name
        learned = None # TBI   
        
        self.settings_model = module.Settings_model()
        self.settings_train = module.Settings_train()
        self.settings_eval = module.Settings_eval()
        self.settings_test = module.Settings_test()
        self.stat_model = module.set_up_model(self.settings_model)          
        

class NeuralNet(Model): # Methods and features shared among all Keras Neural Nets
    def __init__(self,settings):
        super().__init__(settings)

    def setup(self, plot_model=False):
        sys.path.append(config.preset_path)
        module = il.import_module(self.settings.preset) # Import the specified model from file with same name
        learned = None # TBI       
          
        # Record settings from the neural net module    
        self.settings_model = module.Settings_nn()
        self.settings_train = module.Settings_train()
        self.settings_eval = module.Settings_eval()
        self.settings_test = module.Settings_test()
        self.nn = module.set_up_model(self.settings_model)     
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
        
    def set_up_classifier(self):
        sys.path.append(config.classifier_path)
        module = il.import_module(self.settings.classifier)
        self.classifier = module.Classifier(self.classifier_parameters)
            
    def train(self): # Train the neural net
        for example_inputs, example_labels in self.time_series.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')
        tic = time.time()  
        self.history = self.nn.fit(
            self.time_series.train,
            epochs = self.settings_train.epochs,
            batch_size = self.settings_train.batch_size,
            verbose = self.settings_model.verbose)
        self.toc = time.time() - tic
        #self.training_report = self.report_generator.generate_training_report(self)

    def evaluate(self): # Evaluate the neural net
        loss = self.nn.evaluate(
            self.time_series.val,
            batch_size = self.settings_eval.batch_size,
            verbose = self.settings_model.verbose
        )
        print(loss)
        
    def test(self): # Test the neural net
        prediction = self.nn.predict(
            self.time_series.test,
            batch_size = self.settings_test.batch_size,
            verbose = self.settings_model.verbose
        )
        #print(repr(self.time_series))
        shape = tf.shape(prediction)
        ground_truth = tf.concat([y for x, y in self.time_series.test], axis=0)
        self.residual = tf.math.subtract(prediction,ground_truth,name='residual')
     
    def train_classifier_parameters(self):
        prediction = self.nn.predict(
            self.time_series.test,
            batch_size = self.settings_test.batch_size,
            verbose = self.settings_model.verbose
        )
        shape = tf.shape(prediction)
        ground_truth = tf.concat([y for x, y in self.time_series.test], axis=0)
        self.residual = tf.math.subtract(prediction,ground_truth,name='residual')
        tensorized_residual = tf.reshape(self.residual,shape)
        self.classifier_parameters = tf.nn.moments(
            x = tensorized_residual,
            axes = [0]
        )      
        
    def classify(self): # Feed newly fetched data
        prediction = self.nn.predict(
            self.time_series.test
            )
        statistic, pvalue = self.classifier.classify(prediction)
        print(f"t-statistic: {statistic}\npvalue: {pvalue}\n")
        
                
        
    def plot_example(self): # Plot an input-output example
        self.time_series.plot(
            plot_col = self.settings_model.plot_target,
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
            
        #f = open(config.saved_path+self.settings.name+'/report.txt','a')
        #f.write(self.training_report)
        #f.close()
            
            
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
     

     
    

