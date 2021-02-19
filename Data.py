# External packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import rainflow as rf

# Standard packages
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

# Self made modules
from Settings import Settings
import config
import FilterSettings
import RainflowSettings

@dataclass
class DataSeries():
    '''
    This is a dataclass (which requires at leas Python 3.7)
    The attributes are only defined if they are available
    '''
    data : pd.DataFrame(columns = ['placeholder']) # Columns created by get_DataSeries method

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
        
    def filter_data(self,features):
        self.filter_settings = FilterSettings.FilterSettings()
        if self.filter_settings.filter_type == 'lowpass':
            Wn = self.filter_settings.critical_frequency_low 
        elif self.filter_settings.filter_type == 'highpass':
            Wn = self.filter_settings.critiical_frequency_high
        elif self.filter_settings.filter_type in ['bandpass','bandstop']:
            Wn = [self.filter_settings.critical_frequency_low,self.filter_settings.critical_frequency_high]
        else:
            raise Exception('Wrong filter type')
        sos = signal.butter(
            N = self.filter_settings.order,
            Wn = Wn,
            btype = self.filter_settings.filter_type,
            output = 'sos')
        self.filtered = pd.DataFrame(signal.sosfilt(sos,self.data[features]),columns = features)
        
        
            
    def plot_filtered_data(self,features,sensor,start=0,stop=1000):
        fig, ax  = plt.subplots(
            nrows = len(features), # one for each feature
            ncols = 1,
            sharex = True, # They are the same along the x-axis
            sharey = False) # But differ along y
        fig.set_size_inches(config.figsize[0],config.figsize[1])
        for i in range(len(features)): # Loop over features
            ax[i].plot(
                self.data.index[start:stop],
                self.filtered[features[i]][start:stop],
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
        
    def rainflow(self,features,sensor='strain',start=0,stop=-1):
        self.rainflow_settings = RainflowSettings.RainflowSettings()
        fig, ax  = plt.subplots(
            nrows = len(features), # one for each feature
            ncols = 1,
            sharex = True, # They are the same along the x-axis
            sharey = False) # But differ along y
        fig.set_size_inches(config.figsize[0],config.figsize[1])
        rainflows = [rf.count_cycles(self.data[feature][start:stop],binsize=self.rainflow_settings.binsize) 
            for feature in features]
        for i in range(len(rainflows)): # Loop over features
            #print(rainflows[i])
            for j in range(len(rainflows[i])):
                ax[i].plot(rainflows[i][j][0],rainflows[i][j][1],marker = '.',color = 'k')
                #print(rainflows[i][j][0],rainflows[i][j][1])
            #ax[i].bar(np.array(rainflows[i])[0,:],np.array(rainflows[i])[1,:],width = 0.8,align='center')
            ax[i].set_ylabel('Frequency')
            ax[i].grid(alpha = 0.2,zorder = 1)
        plt.xlabel(f'{config.sensors_dict[sensor]}')
        plt.suptitle(f'Rainflow analysis')   
        plt.show()
                    
        
        
class SeriesStack(): # object containing DataSeries objects and meta info
    
    def __init__(self, learned = None, new=True, file_path=config.measurements,
                 sensor='acc',header=0,features=['f0']): # file_path allows for testing
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
        self.sensor = sensor
        self.header = header
        self.features = features
        
    def populate_stack(self,delimiter=';'):
        for i in range(len(self.to_learn)):
            acc = pd.read_table(
                filepath_or_buffer = self.to_learn[i],
                delimiter = delimiter,
                header = self.header, # The header row happens to be on line 22
                names = self.features+['Index'])
            df = pd.DataFrame(acc)
            df['Index'] = df.index
            df.index = range(0,len(df.index))
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1] # move Index to front
            df = df[cols]
            content = self.read_file(self.to_learn[i])
            self.stack.append(self.get_data_series(df, content))
            self.in_stack.update(self.to_learn[i])
            self.stack[i].add_date_to_df()

    def get_data_series(self,df, aux_data):
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
        if self.sensor in ['acc','incl']:
            tstart = aux_data.find('Cut off frequency : ')+20
            COF = aux_data[tstart:tstart+2]
        else: # Strain doesnt have a Cut Off Frequenecy
            COF = 0
        tstart = aux_data.find('Date : ')+7
        DATE = aux_data[tstart:tstart+19]
        data = DataSeries(data,MI,NI,DAC,DAD,SR,COF,DATE)
        return data  
        
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
            
class AccSeriesStack(SeriesStack): # For the Acceleration data
    def __init__(self, learned = None, new=True, file_path=config.measurements,sensor='acc',
                header=22,features=config.acc_features):
        super().__init__(learned = None, new=True, file_path=config.measurements,sensor='acc',
                        header=22,features=config.acc_features)             
    
class InclSeriesStack(SeriesStack): # For the inclination data
    def __init__(self, learned = None, new=True,
                file_path=config.measurements,sensor='incl',header=20,features=config.incl_features):
        super().__init__(learned = None, new=True, file_path=config.measurements,sensor='incl',
                        header=20,features=config.incl_features)         
    
class StrainSeriesStack(SeriesStack): # For the strain data
    def __init__(self, learned = None, new=True, file_path=config.measurements,sensor='strain',
                header=17,features=config.strain_features):
        super().__init__(learned = None, new=True, file_path=config.measurements,sensor='strain',
                        header=17,features=config.strain_features)
            

        
        
        


