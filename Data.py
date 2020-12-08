# External packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Standard packages
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

# Self made modules
from Settings import Settings
import config

@dataclass
class DataBatch():
    '''
    This is a dataclass (which requires at leas Python 3.7)
    The attributes are only defined if they are available
    '''
    data : pd.DataFrame(columns = ['placeholder']) # Columns created by get_databatch method

    # Placeholder values    
    MacId : int = 0
    NetworkId : int = 0
    DataAcqusitionCycle : int = 0
    DataAcqusitionDuration : int = 0
    SamplingRate : int = 0
    CutOffFrequency : int = 0
    Date : int = 0
    
    
    def add_date_to_df(self): # Work in progress
        self.data['date'] = [self.Date] * (self.data.index[-1]+1)
        self.data['date'] = pd.to_datetime(
            self.data['date'],
            dayfirst = False,
            format = config.dateformat)
        self.data['date'] = self.data['date'] + np.arange(0,self.data.index[-1]+1,dtype='timedelta64[s]')
            
    def add_date_signal_to_df(self):
        #print(type(self.data['date'].values[0]))
        day = 24*60*60
        #[]
        self.data['daysignal_sin'] = np.sin(
            (   self.data['date'].hour*3600+self.data['date'].minute*60+
                self.data['date'].second+self.data['date'].microseconds/1e6
                ) * (2 * np.pi / day)
                )
        self.data['daysignal_cos'] = np.cos(
            (   self.data['date'].hour*3600+self.data['date'].minute*60+
                self.data['date'].second+self.data['date'].microseconds/1e6
                ) * (2 * np.pi / day)
                )
        week = 6
        self.data['weeksignal_sin'] = np.sin(
            self.data['date'].weekday() * (2 * np.pi / week)
            )
        self.data['weeksignal_cos'] = np.cos(
            self.data['date'].weekday() * (2 * np.pi / week)
            )
        year = 365.25
        self.data['yearsignal_sin'] = np.sin(
            self.data['date'].timetuple().tm_yday * (2 * np.pi / year)
            )
        self.data['yearsignal_cos'] = np.cos(
            self.data['date'].timetuple().tm_yday * (2 * np.pi / year)
            )
        print(self.data['daysignal_sin'],self.data['daysignal_cos'],self.data['weeksignal_sin'],
        self.data['weeksignal_cos'],self.data['yearsignal_sin'],self.data['yearsignal_cos'])
        
    def plot_data(self,features,sensor,start=0,stop=1000):
        fig, ax  = plt.subplots(
            nrows = len(features), # one for each feature
            ncols = 1,
            sharex = True, # They are the same along the x-axis
            sharey = False) # But differ along y
        fig.set_size_inches(config.figsize[0],config.figsize[1])
        for i in range(len(features)): # Loop over features
            ax[i].plot(
                self.data.index[start:stop],self.data[features[i]][start:stop],
                linewidth = (stop-start)/1000,
                zorder = 2)
            ax[i].set_ylabel(f'{config.sensors_dict[sensor]}')
            ax[i].grid(alpha = 0.2,zorder = 1)
            ax[i].set_title(f'{features[i]}')
        plt.xticks( # Set ticks each second inferred from the sampling rate
            ticks = np.arange(start,stop,self.SamplingRate),
            labels = np.arange(int(start/self.SamplingRate),int(stop/self.SamplingRate)+1,1)
            )
        plt.xlabel('Time[s]')
        plt.suptitle(f'Sensor: {sensor}, Date: {self.Date}, Time step interval: {start}:{stop}')
        plt.show()
        
        
class Series_Stack(): # object containing databatch objects and meta info
    
    def __init__(self, learned = None, new=True, file_path=config.measurements,sensor='acc'): # file_path allows for testing
        '''
        Goes to the location specified by 'file_path' in config.py and set these files as available.
        If the model is reloaded it goes to settings to see which files it already knows.
        The files that are available but not learned are set to be learned.
        '''
        if new:
            self.learned = set() # set() type to make sure there are no duplicates
        else :
            self.learned = learned
        self.available = set()
        months = list(config.months_to_use) #specify the months whose data to use
        for i in range(len(months)): # Count over months
            for j in range(len(config.sensors_folders[sensor])): # Cont over sensors
                try:
                    path = file_path + '/' + months[i] +'/'+ config.sensors_folders[sensor][j]
                    files = os.listdir(path)
                    self.available.update(set(path +'/'+ f for f in files))
                except FileNotFoundError: 
                    pass
        self.to_learn = list(self.available - self.learned)
        self.in_stack = set()
        self.settings = Settings()
        self.stack = list()
        
    def read_file(self,path): 
        ''' 
        Used by populate_stack()
        '''
        with open(path, 'r') as f:
            return f.read()
            
    def pick_series(self,index=None):
        if index == None:
            index = random.randrange(len(self.stack))
        series = list(self.stack)[index]

        return series  
            
class Acc_Series_Stack(Series_Stack): # For the Acceleration data
    def __init__(self, learned = None, new=True, file_path=config.measurements,sensor='acc'):
        super().__init__(learned = None, new=True, file_path=config.measurements,sensor='acc')

    def populate_stack(self,delimiter=';',header=22,features=config.acc_features):
        '''
        Goes to all the 'to_learn' files and records them into DataBatch object with a pd.DataFrame
        '''
        for i in range(len(self.to_learn)):
            acc = pd.read_table(
                filepath_or_buffer = self.to_learn[i],
                delimiter = delimiter,
                header = header, # The header row happens to be on line 22
                names = features+['Index'])
            df = pd.DataFrame(acc)
            df['Index'] = df.index
            df.index = range(0,len(df.index))
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1] # move Index to front
            df = df[cols]
            content = self.read_file(self.to_learn[i])
            self.stack.append(self.get_Databatch(df, content))
            self.in_stack.update(self.to_learn[i])
            self.stack[i].add_date_to_df()
            #print(df.columns)
            
    def get_Databatch(self,df, aux_data):
        '''
        Records the meta data such as sampling rate, which sensor, det etc.
        '''
        data = df
        tstart = aux_data.find('Mac Id : ')+9
        MI = aux_data[tstart:tstart+16]
        tstart = aux_data.find('Network Id : ')+13
        NI = aux_data[tstart:tstart+4]
        tstart = aux_data.find('Data acquisition cycle : ')+25
        DAC = aux_data[tstart:tstart+5]
        tstart = aux_data.find('Data acquisition duration : ')+28
        DAD = aux_data[tstart:tstart+3]
        tstart = aux_data.find('Sampling rate : ')+16
        SR = int(aux_data[tstart:tstart+2])
        tstart = aux_data.find('Cut off frequency : ')+20
        COF = int(aux_data[tstart:tstart+2])
        tstart = aux_data.find('Date : ')+7
        DATE = aux_data[tstart:tstart+19]
        data = DataBatch(data,MI,NI,DAC,DAD,SR,COF,DATE)
        return data    
    
class Incl_Series_Stack(Series_Stack): # For the inclination data
    def __init__(self, learned = None, new=True, file_path=config.measurements,sensor='incl'):
        super().__init__(learned = None, new=True, file_path=config.measurements,sensor='incl')

    def populate_stack(self,delimiter=';',header=20,features=config.incl_features):
        '''
        Goes to all the 'to_learn' files and records them into DataBatch object with a pd.DataFrame
        '''
        for i in range(len(self.to_learn)):
            acc = pd.read_table(
                filepath_or_buffer = self.to_learn[i],
                delimiter = delimiter,
                header = header, # The header row happens to be on line 22
                names = features+['Index'])
            df = pd.DataFrame(acc)
            df['Index'] = df.index
            df.index = range(0,len(df.index))
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1] # move Index to front
            df = df[cols]
            content = self.read_file(self.to_learn[i])
            self.stack.append(self.get_Databatch(df, content))
            self.in_stack.update(self.to_learn[i])
            self.stack[i].add_date_to_df()
            #print(df.columns)
            
    def get_Databatch(self,df, aux_data):
        '''
        Records the meta data such as sampling rate, which sensor, det etc.
        '''
        data = df
        tstart = aux_data.find('Mac Id : ')+9
        MI = aux_data[tstart:tstart+16]
        tstart = aux_data.find('Network Id : ')+13
        NI = aux_data[tstart:tstart+4]
        tstart = aux_data.find('Data acquisition cycle : ')+25
        DAC = aux_data[tstart:tstart+5]
        tstart = aux_data.find('Data acquisition duration : ')+28
        DAD = aux_data[tstart:tstart+3]
        tstart = aux_data.find('Sampling rate : ')+16
        SR = int(aux_data[tstart:tstart+2])
        tstart = aux_data.find('Cut off frequency : ')+20
        COF = aux_data[tstart:tstart+2]
        tstart = aux_data.find('Date : ')+7
        DATE = aux_data[tstart:tstart+19]
        data = DataBatch(data,MI,NI,DAC,DAD,SR,COF,DATE)
        return data  
    
class Strain_Series_Stack(Series_Stack): # For the strain data
    def __init__(self, learned = None, new=True, file_path=config.measurements,sensor='strain'):
        super().__init__(learned = None, new=True, file_path=config.measurements,sensor='strain')

    def populate_stack(self,delimiter=';',header=17,features=config.strain_features):
        '''
        Goes to all the 'to_learn' files and records them into DataBatch object with a pd.DataFrame
        '''
        for i in range(len(self.to_learn)):
            acc = pd.read_table( # read data into a pd.DataFrame
                filepath_or_buffer = self.to_learn[i],
                delimiter = delimiter,
                header = header, # The header row happens to be on line 22
                names = features+['Index'])
            df = pd.DataFrame(acc)
            df['Index'] = df.index
            df.index = range(0,len(df.index))
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1] # move Index to front
            df = df[cols]
            content = self.read_file(self.to_learn[i])
            self.stack.append(self.get_Databatch(df, content))
            self.in_stack.update(self.to_learn[i])
            self.stack[i].add_date_to_df()
            
    def get_Databatch(self,df, aux_data):
        '''
        Records the meta data such as sampling rate, which sensor, det etc.
        '''
        data = df
        tstart = aux_data.find('Mac Id : ')+9
        MI = aux_data[tstart:tstart+16]
        tstart = aux_data.find('Network Id : ')+13
        NI = aux_data[tstart:tstart+4]
        tstart = aux_data.find('Data acquisition cycle : ')+25
        DAC = aux_data[tstart:tstart+5]
        tstart = aux_data.find('Data acquisition duration : ')+28
        DAD = aux_data[tstart:tstart+3]
        tstart = aux_data.find('Sampling rate : ')+16
        SR = int(aux_data[tstart:tstart+2])

        tstart = aux_data.find('Date : ')+7
        DATE = aux_data[tstart:tstart+19]
        data = DataBatch(data,MI,NI,DAC,DAD,SR,0,DATE) # Strain measurements have no cut off frequency
        return data  
        
        
        


