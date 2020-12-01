import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from tensorflow import test
import tensorflow as tf
import os
from datetime import datetime
import copy

import main
import config
from Model import Model, NeuralNet, TimeSeriesNeuralNet
from Settings import Settings
from Menu import *
from functions import *
from Data import *
#import functions




'''    
class <classname>_test(unittest.TestCase):

    @unittest.skip('')
    def test_<method_name>(self):
        set things up       
        self.assert(something)
'''

'''
def boilerplate_function():
    repetitive stuff
'''
    
class TimeSeriesNeuralNet_test(unittest.TestCase):

    def test_make_time_series_dataset(self):
        model = make_model(preset='RNN_test')
        model.make_timeseries_dataset(print_shape=True)
        for example_inputs, example_labels in model.time_series.train.take(1):
            pass
        self.assertEqual(
            example_inputs.shape,
            ([model.settings_train.batch_size,
              model.settings_nn.input_time_steps,
              len(model.settings_nn.features)]))
        self.assertEqual(
            example_labels.shape,
            ([model.settings_train.batch_size,
              model.settings_nn.target_time_steps,
              len(model.settings_nn.features)]))
              
        for example_inputs, example_labels in model.time_series.val.take(1):
            pass
        self.assertEqual(
            example_inputs.shape,
            ([model.settings_eval.batch_size,
              model.settings_nn.input_time_steps,
              len(model.settings_nn.features)]))
        self.assertEqual(
            example_labels.shape,
            ([model.settings_eval.batch_size,
              model.settings_nn.target_time_steps,
              len(model.settings_nn.features)]))    
              
        for example_inputs, example_labels in model.time_series.test.take(1):
            pass
        self.assertEqual(
            example_inputs.shape,
            ([model.settings_test.batch_size,
              model.settings_nn.input_time_steps,
              len(model.settings_nn.features)]))
        self.assertEqual(
            example_labels.shape,
            ([model.settings_test.batch_size,
              model.settings_nn.target_time_steps,
              len(model.settings_nn.features)]))   
              
    def test_predict(self):
        model = make_model(preset = 'RNN_test',file_path = config.test_measurements)
        model.make_timeseries_dataset()
        model.train()
        model.predict()    
        
class RNN_single_step_test(unittest.TestCase): 
    
    def test_setup(self):
        model = make_model(preset='RNN_single_step')
        model.make_timeseries_dataset()
        model.train()
        model.evaluate()      
        
class RNN_multi_step_test(unittest.TestCase): 
    
    def test_setup(self):
        model = make_model(preset='RNN_multi_step')
        model.make_timeseries_dataset()
        model.train()
        model.evaluate()                                         
            
class SLP_single_step_test(unittest.TestCase):
    
    def test_setup(self):
        model = make_model(preset='SLP_single_step') 
        model.make_timeseries_dataset()
        model.train()
        model.evaluate()  
        
class SLP_multi_step_test(unittest.TestCase):
    
    def test_setup(self):
        model = make_model(preset='SLP_multi_step') 
        model.make_timeseries_dataset()
        model.train()
        model.evaluate()  
        
def make_model(preset=None,file_path = config.test_measurements):
    settings = Settings()
    if preset is not None:
        settings.preset = preset
    model = TimeSeriesNeuralNet(settings,False)
    learned = model.setup_nn()
    series_stack = Series_Stack(settings,'new',file_path = file_path)
    series_stack.populate_stack()
    model.make_dataframe(series_stack)
    return model
    

if __name__ == '__main__':
    #file1 = config.test_measurements+'/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
    #file2 = config.test_measurements+'/aug 2020/Acc2/'+'Transmit_Streaming_MacId_00158D00000E0FE9_2020_08_01_02_34_42.txt'
    unittest.main()


