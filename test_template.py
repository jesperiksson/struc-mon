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
    
 
    

if __name__ == '__main__':
    #file1 = config.test_measurements+'/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
    #file2 = config.test_measurements+'/aug 2020/Acc2/'+'Transmit_Streaming_MacId_00158D00000E0FE9_2020_08_01_02_34_42.txt'
    unittest.main()


