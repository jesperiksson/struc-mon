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

    @unittest.skip('redundant')
    def test_test(self):
        model = make_model(preset = 'SLP_multi_step',file_path = config.train_test_measurements)
        model.make_timeseries_dataset()
        #model.train()
        #model.save_nn()
        model.load_nn()
        model.test()
        
    def test_detect_outliers(self):
        model = make_model(preset = 'SLP_multi_step',file_path = config.train_test_measurements)
        model.make_timeseries_dataset()
        #model.train()
        #model.save_nn()
        model.load_nn()
        model.test()
        model.detect_outliers() 
        model.plot_outliers()             

def make_model(preset=None,file_path = config.test_measurements):
    settings = Settings()
    if preset is not None:
        settings.preset = preset
    model = TimeSeriesNeuralNet(settings,False)
    learned = model.setup_nn(print_summary=False,plot_model=True)
    series_stack = Series_Stack(settings,'new',file_path = file_path)
    series_stack.populate_stack()
    model.make_dataframe(series_stack)
    return model
    

if __name__ == '__main__':
    #file1 = config.test_measurements+'/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
    #file2 = config.test_measurements+'/aug 2020/Acc2/'+'Transmit_Streaming_MacId_00158D00000E0FE9_2020_08_01_02_34_42.txt'
    unittest.main()


