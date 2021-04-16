# Standard packages
import time
import importlib as il
import sys
import os
import pickle
import random

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
#from WindowClassificationGenerator import *
from ReportGenerator import ReportGenerator

tf.get_logger().setLevel('ERROR')

class Model(): # Methods and features shared across all predictive models 
    def __init__(self,settings):
        
        self.settings = settings
        #self.name = settings.name
        self.report_generator = ReportGenerator(settings)

        
    def detect_outliers(self): # Calculates p-value for residual
        statistic , pvalue = stats.normaltest(self.residual)
        np.set_printoptions(precision=4)
        
    def plot_outliers(self): # Plots histogram of predictions an a scatterplot
        fig, (ax1,ax2)  = plt.subplots(nrows=1,ncols=2,figsize=config.figsize)
        ax1.hist(
            self.residual.numpy()[:,0,0],
            bins = 100)
        ax1.set_title('Prediction residuals')
        ax1.set_ylabel(f'Prediction error for {self.settings_model.plot_targets}')
        ax2.hist(
            self.prediction.flatten(),
            bins = 100)
        ax2.set_title('Predictions')
        plt.show()
        
    def remember_dates(self,data):
        return {
            'train_start' : data.train_df['ts'].iloc[0],
            'train_end' : data.train_df['ts'].iloc[-1],
            'test_start' : data.test_df['ts'].iloc[0],
            'test_end' : data.test_df['ts'].iloc[-1],
            'eval_start' : data.val_df['ts'].iloc[0],
            'eval_end' : data.val_df['ts'].iloc[-1],
        }
        
    def print_shape(self):   
        rand = random.randint(0,1000000)
        try:
            for example_inputs, example_labels in self.time_seriess[rand%len(self.time_seriess)].train.take(1):
                print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
                print(f'Labels shape (batch, time, features): {example_labels.shape}')
        except ValueError:
            for example_inputs, example_labels in self.time_seriess[rand%len(self.time_seriess)].train.test(1):
                print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
                print(f'Labels shape (batch, time, features): {example_labels.shape}')
                
    def save_dataset(self):
        with open(f"{config.dataset_path}{self.settings.dataset_name}.json",'wb') as f:
            pickle.dump(self.time_series, f)
            
    def load_dataset(self):
        self.time_series = pickle.load(open(f"{config.dataset_path}{self.settings.dataset_name}.json",'rb'))
        
    def inspect_dataset(self):
        print(len(list(self.time_series.train.as_numpy_iterator())))    
        
class StatModel(Model): # Purely statistical model to be used as baseline
    def __init__(self,settings,data):
        super().__init__(settings)
        self.data = data
        
    def setup(self):
        sys.path.append(config.preset_path)
        module = il.import_module(self.settings.preset) # Import the specified model from file with same name
        
        self.model = module.Model(self.data)   
             
        

class NeuralNet(Model): # Methods and features shared among all Keras Neural Nets
    def __init__(self,settings):
        super().__init__(settings)
        
    def make_timeseries_dataset(self, data):
        time_seriess = []
        cols = list(set(self.settings_model.features+self.settings_model.targets))
        for i in range(len(data.dfs)):      
            time_seriess.append(WindowGenerator(
                input_width = self.settings_model.input_time_steps,
                label_width = self.settings_model.target_time_steps,
                shift = self.settings_model.shift,
                train_df = data.train_dfs[i][cols],
                val_df = data.val_dfs[i][cols],
                test_df = data.test_dfs[i][cols],
                feature_columns = self.settings_model.features,
                label_columns = self.settings_model.targets,
                train_batch_size = self.settings_train.batch_size,
                eval_batch_size = self.settings_eval.batch_size,
                test_batch_size = self.settings_test.batch_size)
                )
            
        self.time_seriess = time_seriess
        try:
            self.dates = data.dates
        except AttributeError:
            self.dates = 'missing'
            

    def setup(self):
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
        
    def compile_model(self):
        self.nn.compile( # compile the net
            loss = self.settings_train.loss,
            optimizer = self.settings_train.optimizer,
            metrics = self.settings_train.metrics)
        self.loaded = False   
               
    def plot_model(self):
        tf.keras.utils.plot_model(
            self.nn, 
            config.saved_path+self.settings.name+'/model_plot.jpeg',
            show_shapes = True,
            show_layer_names = True,
            dpi = 150)
            
        
    def set_up_classifier(self):
        sys.path.append(config.classifier_path)
        module = il.import_module(self.settings.classifier)
        self.classifier = module.Classifier(self.classifier_parameters)
            
    def train(self): # Train the neural net
        train = self.time_seriess[0].train
        val = self.time_seriess[0].val
        for time_series in self.time_seriess:
            train = train.concatenate(time_series.train)
            val = val.concatenate(time_series.val)
        for example_inputs, example_labels in time_series.train.take(1):
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
            train,
            epochs = self.settings_train.epochs,
            batch_size = self.settings_train.batch_size,
            validation_data = val,
            callbacks = callbacks,
            shuffle = self.settings_train.shuffle,
            verbose = self.settings_model.verbose)
        self.toc = time.time() - tic
        if self.loaded:
            for key in self.history.history.keys():
                self.history.history[key].extend(self.earlier_history[key])
        self.nn.summary()
        
        self.training_report = self.report_generator.generate_training_report(self)

    def evaluate(self): # Evaluate the neural net
        test = self.time_seriess[0].test
        for time_series in self.time_seriess:
            test = test.concatenate(time_series.test)
        self.loss = self.nn.evaluate(
            test,
            batch_size = self.settings_eval.batch_size,
            verbose = self.settings_model.verbose
        )
        self.eval_report = self.report_generator.generate_eval_report(self)
        
    def test(self): # Test the neural net
        test = self.time_seriess[0].test
        for time_series in self.time_seriess:
            test = test.concatenate(time_series.test)
        prediction = self.nn.predict(
            test,
            batch_size = self.settings_test.batch_size,
            verbose = self.settings_model.verbose
        )
        #print(repr(time_series))
        shape = tf.shape(prediction)
        ground_truth = tf.concat([y for x, y in test], axis=0)
        self.residual = tf.math.subtract(prediction,ground_truth,name='residual')
        self.prediction = prediction
     
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
                    
           
    def save_nn(self,overwrite=False):
        name = self.get_name()
        path = config.saved_path+name 
        self.nn.save(
            filepath = path,
            overwrite = overwrite,
            include_optimizer = True,
            save_format = 'tf')
            
        with open(path+'/history.json','wb') as f:
            pickle.dump(self.history.history, f)
        f = open(config.saved_path+name+'/train_report.txt','a')
        f.write(self.training_report)
        f.close()
        f = open(config.saved_path+name+'/eval_report.txt','a')
        f.write(self.eval_report)
        f.close()
            
    def load_nn(self):
        name = self.get_name()
        if name not in os.listdir(config.saved_path):
            print(f'No saved model named {name}')
            #raise Exception('No module to load')
        else:
            print(config.saved_path+name,os.listdir(config.saved_path))
            loaded_nn = tf.keras.models.load_model(
                filepath = config.saved_path+name,
                compile = True)
            self.nn = loaded_nn
            print(f"Loaded {name}")

        self.earlier_history = pickle.load(open(config.saved_path+name+'/history.json','rb'))
        self.loaded = True
        
    def get_name(self):
        s = self.settings_model
        return f"{s.kind}_{s.input_time_steps}_{s.target_time_steps}_{s.shift}_nodes_{'_'.join([str(x) for x in s.layer_widths])}_in_{'-'.join(s.features)}_out_{'-'.join(s.targets)}"

        
class TimeSeriesPredictionNeuralNet(NeuralNet):
    def __init__(self,settings):
        super().__init__(settings)
        

        
    def plot_history(self): # Plot the training history for each metric
        key_list = list(self.history.history.keys())
        [plt.plot(self.history.history[key]) for key in key_list]
        plt.legend(key_list)
        plt.title(f'Training history for {self.get_name}, trained for {self.settings_train.epochs} epochs. Elaspsed time: {self.toc}')
        plt.xlabel('epoch')
        plt.ylabel('error') 
        plt.savefig(config.saved_path+self.get_name()+''.join(self.settings.sensors))
        plt.show()   
        
    def plot_example(self): # Plot an input-output example
        rand = random.randint(0,1000000)
        self.time_seriess[rand%len(self.time_seriess)].plot(
            plot_cols = self.settings_model.plot_targets,
            model = self.nn)
        plt.show()  
        
class TimeSeriesClassificationNeuralNet(NeuralNet): 
    def __init__(self,settings):
        super().__init__(settings)
        
    def make_timeseries_category_dataset(self, data):
        time_seriess = []
        cols = sorted(list(set(self.settings_model.features+self.settings_model.targets)))
        for i in range(len(data.dfs)):      
            time_seriess.append(
                WindowClassificationGenerator(
                    input_width = self.settings_model.input_time_steps,
                    shift = self.settings_model.shift,
                    train_df = data.train_dfs[i][cols],
                    val_df = data.val_dfs[i][cols],
                    test_df = data.test_dfs[i][cols],
                    train_batch_size = self.settings_train.batch_size,
                    eval_batch_size = self.settings_eval.batch_size,
                    test_batch_size = self.settings_test.batch_size)
                )
            
        self.time_seriess = time_seriess
        self.dates = data.dates
        
    def plot_history(self): # Plot the training history for each metric
        key_list = ['binary_crossentropy','accuracy','auc']
        [plt.plot(self.history.history[key]) for key in key_list]
        plt.legend(key_list)
        plt.title(f'Training history for {self.get_name()}, trained for {self.settings_train.epochs} epochs. Elaspsed time: {self.toc}')
        plt.xlabel('epoch')
        plt.ylabel('error') 
        plt.savefig(config.saved_path+self.settings.name+''.join(self.settings.sensors))
        plt.show()   
        
    def plot_auc(self):
        plt.plot(
#            self.history.history['auc'],
#            self.history.history['false_positives']*(1/max(self.history.history['false_positives'])),
#            self.history.history['true_positives']*(1/max(self.history.history['true_positives'])),
            [x/max(self.history.history['false_positives']) for x in self.history.history['false_positives']],
            [x/max(self.history.history['true_positives']) for x in self.history.history['true_positives']],
            color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

     

     
    

