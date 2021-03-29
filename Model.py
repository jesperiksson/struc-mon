# Standard packages
import time
import importlib as il
import sys
import os
import pickle

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
        plt.ylabel(f'Prediction error for {self.settings_model.plot_targets}')
        plt.show()
        
    def make_timeseries_dataset(self, data, print_shape=False):
        time_seriess = []
        for i in range(len(data.dfs)):      
            time_seriess.append(WindowGenerator(
                input_width = self.settings_model.input_time_steps,
                label_width = self.settings_model.target_time_steps,
                shift = self.settings_model.shift,
                train_df = data.train_dfs[i][self.settings_model.features],
                val_df = data.val_dfs[i][self.settings_model.features],
                test_df = data.test_dfs[i][self.settings_model.features],
                label_columns = self.settings_model.targets,
                train_batch_size = self.settings_train.batch_size,
                eval_batch_size = self.settings_eval.batch_size,
                test_batch_size = self.settings_test.batch_size)
                )
        self.time_series = time_seriess[0]
        for i in range(len(time_seriess)-1):
            self.time_series.train.concatenate(time_seriess[i+1].train)
            self.time_series.test.concatenate(time_seriess[i+1].test)
            self.time_series.val.concatenate(time_seriess[i+1].val)
        try:
            if print_shape:
                for example_inputs, example_labels in self.time_series.train.take(1):
                    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
                    print(f'Labels shape (batch, time, features): {example_labels.shape}')
        except ValueError:
            for example_inputs, example_labels in self.time_series.train.test(1):
                print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
                print(f'Labels shape (batch, time, features): {example_labels.shape}')
                
    def save_dataset(self):
        with open(f"{config.dataset_path}{self.settings.dataset_name}.json",'wb') as f:
            pickle.dump(self.time_series, f)
            
    def load_dataset(self):
        self.time_series = pickle.load(open(f"{config.dataset_path}{self.settings.dataset_name}.json",'rb'))
        
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
        self.loaded = False
        
    def set_up_classifier(self):
        sys.path.append(config.classifier_path)
        module = il.import_module(self.settings.classifier)
        self.classifier = module.Classifier(self.classifier_parameters)
            
    def train(self): # Train the neural net
        for example_inputs, example_labels in self.time_series.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')
        tic = time.time()
        callbacks = []
        if self.settings_train.early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor = self.settings_train.early_stopping_monitor,
                min_delta  = self.settings_train.early_stopping_min_delta,
                patience = self.settings_train.early_stopping_patience,
                verbose = self.settings_train.early_stopping_verbose,
                mode = self.settings_train.early_stopping_mode
            ))
        self.history = self.nn.fit(
            self.time_series.train,
            epochs = self.settings_train.epochs,
            batch_size = self.settings_train.batch_size,
            validation_data = self.time_series.val,
            callbacks = callbacks,
            verbose = self.settings_model.verbose)
        self.toc = time.time() - tic
        if self.loaded:
            for key in self.history.history.keys():
                self.history.history[key].extend(self.earlier_history[key])
        self.nn.summary()
        
        #self.training_report = self.report_generator.generate_training_report(self)

    def evaluate(self): # Evaluate the neural net
        loss = self.nn.evaluate(
            self.time_series.val,
            batch_size = self.settings_eval.batch_size,
            verbose = self.settings_model.verbose
        )
        
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
        # Write a report 
              
    def plot_example(self): # Plot an input-output example
        self.time_series.plot(
            plot_cols = self.settings_model.plot_targets,
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
        print(self.nn)
        self.nn.save(
            filepath = path,
            overwrite = overwrite,
            include_optimizer = True,
            save_format = 'tf')
            
        with open(path+'/history.json','wb') as f:
            pickle.dump(self.history.history, f)
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

        self.earlier_history = pickle.load(open(config.saved_path+self.settings.name+'/history.json','rb'))
        self.loaded = True

        
class TimeSeriesNeuralNet(NeuralNet): # For RNNs, CNNs, etc. Obsolete?
    def __init__(self,settings):
        super().__init__(settings)
     

     
    

