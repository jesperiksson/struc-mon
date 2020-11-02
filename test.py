import unittest
from mock import patch
import numpy as np
import pandas as pd
import os

import main
import config
import NeuralNet
from set_settings import set_settings
from Menu import *
from functions import *
from Data import *
#import functions

class Test_class(unittest.TestCase):
    # test functions must contain 'test' in the beginning of name
    def test_func_1(self):
        pass

class Menu_test(unittest.TestCase):
    
    def test_quit(self):
        with self.assertRaises(SystemExit):
            quit()

    #def test_menu(self):

class functions_test(unittest.TestCase):
    

    @unittest.skip('Too much to print')
    def test_read_file(self):       
        self.assertEqual(
            read_file(file1),
            'foo'
            )
    
    '''
    def test_get_Databatch(self):
        data, learned = get_data_one_by_one({file1})
        self.assertIsInstance(data, Databatch.DataBatch)'''

class DataBatch_test(unittest.TestCase):
    pass
    
class Series_Stack_test(unittest.TestCase):
  

    def test_init(self):
        settings = set_settings(placeholder=True)
        file1 = '/home/jesper/Kurser/Exjobb/ANN/code/files/test_measurements/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
        file2 = '/home/jesper/Kurser/Exjobb/ANN/code/files/test_measurements/aug 2020/Acc2/'+'Transmit_Streaming_MacId_00158D00000E0FE9_2020_08_01_02_34_42.txt'
        settings.update({'learned' : set([file1])})
        Stack = Series_Stack(settings,'old')  
        self.assertEqual(Stack.learned,set([file1]))
        self.assertEqual(Stack.to_learn,list([file2]))
        self.assertEqual(Stack.available,set([file1,file2]))
        
        Stack.populate_stack()
        self.assertIsInstance(Stack.stack[0],DataBatch)
        
    def test_read_file(self):
        self.assertEqual(
            Series_Stack.read_file(self,'/home/jesper/Kurser/Exjobb/ANN/code/files/dummyfiles/dummy.txt'),
            'foobarz\n')
        
    def test_get_Databatch(self):
        x1 = np.random.rand(10,1)
        y1 = np.random.rand(10,1)
        z1 = np.random.rand(10,1)
        heads = ['x','y','z']
        data = np.concatenate([x1,y1,z1],axis=-1)
        content = Series_Stack.read_file(self,'/home/jesper/Kurser/Exjobb/ANN/code/files/test_measurements/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt')
        batch = Series_Stack.get_Databatch(self,pd.DataFrame(data,columns=heads), content)
        self.assertIsInstance(
            batch,
            DataBatch)
        self.assertEqual(batch.MacId, '00158D00000E054C')  
     
    @unittest.skip('No return')          
    def test_populate_stack(self):
        # This method has no return
        pass
    
class NeuralNet_test(unittest.TestCase):

    @unittest.skip('Not ready')
    def test_model(self):
        file1 = '/home/jesper/Kurser/Exjobb/ANN/code/files/test_measurements/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
        model = NeuralNet(
            settings = set_settings(),
            existing_model = False)
        data, learned = get_data_one_by_one({file1})
        self.assertIsInstance(model,NeuralNet)
    @unittest.skip('Not ready')  
    def test_data_splitter(self):
        file1 = '/home/jesper/Kurser/Exjobb/ANN/code/files/test_measurements/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
        data, learned = get_data_one_by_one({file1})
        model = NeuralNet(
            settings = set_settings(),
            existing_model = False)
        self.assertEqual(NeuralNet.data_splitter(model,data,['']),[])
    
    '''
    def test_train(self):
        file1 = '/home/jesper/Kurser/Exjobb/ANN/code/files/test_measurements/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
        model = NeuralNet(
            settings = set_settings(),
            existing_model = False)
        data, learned = get_data_one_by_one({file1})
        model.train(data)
    pass'''
        
class set_settings_test(unittest.TestCase):
    def test_set_settings(self):
        self.assertEqual(set_settings(placeholder = True), {'name' : 'placeholder'})

if __name__ == '__main__':
    file1 = '/home/jesper/Kurser/Exjobb/ANN/code/files/test_measurements/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
    file2 = '/home/jesper/Kurser/Exjobb/ANN/code/files/test_measurements/aug 2020/Acc2/'+'Transmit_Streaming_MacId_00158D00000E0FE9_2020_08_01_02_34_42.txt'
    import config
    unittest.main()


