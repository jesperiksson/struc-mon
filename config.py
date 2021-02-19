file_path = '/home/jesper/Kurser/Exjobb/ANN/code/files'


measurements = file_path+'/measurements'
test_measurements = file_path+'/test_measurements' # Remove after testing
train_test_measurements = file_path+'/train_test_measurements'

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
preset_path = dir_path + '/presets/'
template_path = dir_path + '/templates/'
saved_path = dir_path + '/saved/'

months = {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}

months_to_use = {'sep 2020 test'}

NetworkId = {
    '0002' : 'Acc1', 
    '0001' : 'Acc2', 
    '0005' : 'Incl', 
    '0006' : 'Strain',
    '0004' : 'Temp'
    }
    
acc_features = ['x','y','z']
incl_features = ['x','y']
strain_features = ['ch0','ch1','ch2','ch3']

    
sensors_dict = {
    'acc' : 'Acceleration[g]',
    'incl' : 'Inclination[deg]',
    'strain' : 'Strain[mV]'
}

sensors_folders = {
    'acc' : ['Acc1','Acc2'],
    'incl' : ['Incl'],
    'strain' : ['Strain']
}

MacId = {
    'Acc1' : '00158D00000E054C',
    'Acc2' : '00158D00000E0FE9',
    'Incl' : '00158D00000E1024',
    'Strain':'00158D00000E0EE5',
    'Temp' : '00158D00000E047B'
    }
    
dateformat = '%Y-%m-%d %H:%M:%S'
    
figsize = [12,8] # [width, height]   
 
