




import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
preset_path = dir_path + '/presets/'
#template_path = dir_path + '/templates/'
saved_path = dir_path + '/saved/'


SQLA_GOST_DATABASE_USER = input('Username: ')
SQLA_GOST_DATABASE_PASS = input('Password: ')
SQLA_GOST_DATABASE_HOST = 'localhost'
SQLA_GOST_DATABASE_PORT = '5432'
SQLA_GOST_DATABASE_NAME = 'gost'

#TFIO_GOST_DATABASE_NAME='gost'
#TFIO_GOST_DATABASE_HOST='127.0.0.1:5050'
#TFIO_GOST_DATABASE_PORT='5432'
#TFIO_GOST_DATABASE_USER=''
#TFIO_GOST_DATABASE_PASS=

table_names = {
    'strain1' : 'sensor_00158d00000e0ee5',
    'acc1' : 'sensor_00158d00000e0fe9',
    'acc2' : 'sensor_00158d00000e054c',
    'incl' : 'sensor_00158d00000e1024',
    'temp' : 'sensor_00158d00000e047b',
    'strain2': 'sensor_000000008bff4366'
}

time_stamp = 'ts'
column_names = {
    table_names['strain1'] : ['id',time_stamp,'ch_mv0','ch_mv1','ch_mv2','ch_mv3'],
    table_names['acc1'] : ['id',time_stamp,'ch_x','ch_y','ch_z'],
    table_names['acc2'] : ['id',time_stamp,'ch_x','ch_y','ch_z'],
    table_names['incl'] : ['id',time_stamp,'ch_x','ch_y'],
    table_names['temp'] : ['id',time_stamp,'ch_temperature'],
    table_names['strain2'] : ['id',time_stamp,'ch_mv0','ch_mv0_379']
}

schema = 'v1'


MacId = {
    'Acc1' : '00158D00000E054C',
    'Acc2' : '00158D00000E0FE9',
    'Incl' : '00158D00000E1024',
    'Strain':'00158D00000E0EE5',
    'Temp' : '00158D00000E047B'
    }
    
dateformat = '%Y-%m-%d' #%H:%M:%S'
reg_dateformat = '\d\d\d\d[_]\d\d[_]\d\d[_]\d\d[_]\d\d[_]\d\d'


    
figsize = [12,8] # [width, height]   
 
