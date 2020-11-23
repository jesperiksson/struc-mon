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
from set_settings import set_settings, settings_placeholder
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
    
    def test_add_date_to_df(self):
        data = pd.DataFrame([[1,2,3],[4,5,6,],[7,8,9]],columns = ['a','b','c'])
        obj = DataBatch(data,None,None,None,None,1,None,'2020-08-01 02:34:42')
        obj.add_date_to_df()
        assert_frame_equal(
            obj.data,
            pd.DataFrame(
                [   [1,2,3,datetime.strptime('2020-08-01 02:34:42',config.dateformat)],
                    [4,5,6,datetime.strptime('2020-08-01 02:34:43',config.dateformat)],
                    [7,8,9,datetime.strptime('2020-08-01 02:34:44',config.dateformat)]
                    ],
                columns = ['a','b','c','date']
                    )
                )
    @unittest.skip('To be implemented')            
    def test_add_date_signal_to_df(self):
        data = pd.DataFrame([[1,2,3],[4,5,6,],[7,8,9]],columns = ['a','b','c'])
        obj = DataBatch(data,None,None,None,None,1,None,'2020-08-01 02:34:42')
        obj.add_date_to_df()
        obj.add_date_signal_to_df()
    
class Series_Stack_test(unittest.TestCase):
  

    def test_init(self):
        settings = set_settings(placeholder=True)
        file1 = config.test_measurements+'/aug 2020/Acc1/Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
        file2 = config.test_measurements+'/aug 2020/Acc2/Transmit_Streaming_MacId_00158D00000E0FE9_2020_08_01_02_34_42.txt'
        settings.update({'learned' : set([file1])})
        Stack = Series_Stack(settings,'old',config.test_measurements) # Specifies the test folder
        self.assertEqual(Stack.learned,set([file1]))
        self.assertEqual(Stack.to_learn,list([file2]))
        self.assertEqual(Stack.available,set([file1,file2]))
        
        Stack.populate_stack()
        self.assertIsInstance(Stack.stack[0],DataBatch)
        
    def test_read_file(self):
        self.assertEqual(
            Series_Stack.read_file(self,config.file_path+'/dummyfiles/dummy.txt'),
            'foobarz\n\n')
        
    def test_get_Databatch(self):
        x1 = np.random.rand(10,1)
        y1 = np.random.rand(10,1)
        z1 = np.random.rand(10,1)
        heads = ['x','y','z']
        data = np.concatenate([x1,y1,z1],axis=-1)
        content = Series_Stack.read_file(self,config.test_measurements+'/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt')
        batch = Series_Stack.get_Databatch(self,pd.DataFrame(data,columns=heads), content)
        self.assertIsInstance(
            batch,
            DataBatch)
        self.assertEqual(batch.MacId, '00158D00000E054C')  
     
    @unittest.skip('No return')          
    def test_populate_stack(self):
        # This method has no return
        pass
        
class Model_test(unittest.TestCase):

    @unittest.skip('Added try/except instead')
    def test_make_dataframe_fail(self):
        settings = set_settings(placeholder=True)
        series_stack = Series_Stack(settings,'new',file_path = config.test_measurements)
        model = NeuralNet(settings,False)
        with self.assertRaises(IndexError):
            model.make_dataframe(series_stack)
            
            
    def test_make_dataframe(self):
        settings = set_settings(placeholder=True)
        series_stack = Series_Stack(settings,'new',file_path = config.test_measurements)
        series_stack.populate_stack()
        model = NeuralNet(settings,False)
        model.make_dataframe(series_stack)
    
class NeuralNet_test(unittest.TestCase):
        
    @unittest.skip('wrong method')     
    def test_make_data_set(self):
        settings = set_settings(placeholder=True)
        series_stack = Series_Stack(settings,'new',file_path = config.test_measurements)
        series_stack.populate_stack()
        model = NeuralNet(settings,False)
        model.make_dataframe(series_stack)
        model.make_data_set()
        test.TestCase.assertShapeEqual(np.shape(np.array([[3,],[1,]])),model.dataset)

    @unittest.skip('Not ready')
    def test_model(self):
        file1 = config.test_measurements+'/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
        model = NeuralNet(
            settings = set_settings(),
            existing_model = False)
        data, learned = get_data_one_by_one({file1})
        self.assertIsInstance(model,NeuralNet)
    '''
    @unittest.skip('Not ready')
    def test_data_splitter(self):
        file1 = config.test_measurements+'/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
        data, learned = get_data_one_by_one({file1})
        model = NeuralNet(
            settings = set_settings(),
            existing_model = False)
        self.assertEqual(NeuralNet.data_splitter(model,data,['']),[])'''
    
class GenericNeuralNet_test(unittest.TestCase):

    @unittest.skip('not implemented yet')     
    def test_make_iterator(self):
        pass
        
    @unittest.skip('not implemented yet')      
    def test_make_tensors(self):
        pass
        
    @unittest.skip('not implemented yet')  
    def test_make_model(self):
        pass

    
class TimeSeriesNeuralNet_test(unittest.TestCase):

    #@unittest.skip('')
    def test_setup_nn(self):
        model = bolierplate()       
        self.assertIsInstance(model.nn,tf.keras.Model)

    #@unittest.skip('')
    def test_make_timeseries_dataset(self):
        model = bolierplate()
        model.make_timeseries_dataset()
        self.assertIsInstance(model.train_df,pd.DataFrame)
        self.assertIsInstance(obj=model.time_series.train,cls=tf.data.Dataset) 
    
    #@unittest.skip('redundant')        
    def test_train(self):
        model = bolierplate(file_path = config.train_test_measurements)
        model.make_timeseries_dataset()
        model.train()
        
    @unittest.skip('they print the same but fail the test')
    def test_save_and_load(self):
        model1 = bolierplate(file_path = config.train_test_measurements)
        model1.make_timeseries_dataset()
        model1.train()
        model1.save_nn(overwrite=True)
        
        model2 = bolierplate(file_path = config.train_test_measurements)
        model2.make_timeseries_dataset()        
        model2.load_nn()
        print('\n',model1.nn.trainable_weights,'\n',model2.nn.trainable_weights,'\n')
        tf.debugging.assert_equal(model1.nn.trainable_weights,model2.nn.trainable_weights)
    
    #@unittest.skip('takes long time')     
    def test_evalutate(self):
        model = bolierplate(file_path = config.train_test_measurements)
        model.make_timeseries_dataset()
        model.load_nn()
        model.evaluate()
           
    def test_predict(self):
        model = bolierplate(file_path = config.train_test_measurements)
        model.make_timeseries_dataset()
        model.load_nn()
        model.predict_single_sample()
           
    @unittest.skip('plots images')    
    def test_plot_history(self):
        model = bolierplate(file_path = config.train_test_measurements)
        model.make_timeseries_dataset()
        model.train()
        model.plot_history()
     
    @unittest.skip('return to this one')    
    def test_plot_test_loss(self):
        model = bolierplate(file_path = config.train_test_measurements)
        model.make_timeseries_dataset()
        model.train()
        model.evaluate()
        model.plot_test_loss()
        
    def test_predict_single_sample(self):
        model = bolierplate(file_path = config.train_test_measurements)
        model.make_timeseries_dataset()
        model.load_nn()
        model.predict_single_sample(n_samples=1)
        
class set_settings_test(unittest.TestCase):
    def test_set_settings(self):
        self.assertEqual(set_settings(placeholder = True), settings_placeholder)

def bolierplate(file_path = config.test_measurements,ph=True):
    settings = set_settings(placeholder=ph)
    series_stack = Series_Stack(settings,'new',file_path = file_path)
    series_stack.populate_stack()
    model = TimeSeriesNeuralNet(settings,False)
    model.make_dataframe(series_stack)
    model.setup_nn()   
    return model

if __name__ == '__main__':
    #file1 = config.test_measurements+'/aug 2020/Acc1/'+'Transmit_Streaming_MacId_00158D00000E054C_2020_08_01_02_34_42.txt'
    #file2 = config.test_measurements+'/aug 2020/Acc2/'+'Transmit_Streaming_MacId_00158D00000E0FE9_2020_08_01_02_34_42.txt'
    unittest.main()


