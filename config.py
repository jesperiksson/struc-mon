file_path = '/home/jesper/Kurser/Exjobb/ANN/code/files'


measurements = file_path+'/measurements'
test_measurements = file_path+'/test_measurements' # Remove after testing
train_test_measurements = file_path+'/train_test_measurements'

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = dir_path + '/models/'

months = {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}

months_to_use = {'aug 2020'}

NetworkId = {
    '0002' : 'Acc1', 
    '0001' : 'Acc2', 
    '0005' : 'Incl', 
    '0006' : 'Strain',
    '0004' : 'Temp'
    }
    
sensors_of_interest = {
    0 : 'Acc1', 
    1 : 'Acc2'
    }

MacId = {
    'Acc1' : '00158D00000E054C',
    'Acc2' : '00158D00000E0FE9',
    'Incl' : '00158D00000E1024',
    'Strain':'00158D00000E0EE5',
    'Temp' : '00158D00000E047B'
    }
    
dateformat = '%Y-%m-%d %H:%M:%S'
    
