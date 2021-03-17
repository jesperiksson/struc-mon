# Standard packages
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# External packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scipy.signal as signal
import tensorflow.signal as signal
import rainflow as rf
import seaborn as sns



# Self made modules
from Settings import Settings
import config
#import FilterSettings
#import RainflowSettings

class Data():
    def __init__(self,query_generator,connection):
        self.query_generator = query_generator
        self.connection = connection
        
    def make_df_postgres(self):
        self.df = pd.read_sql_query(
            sql = self.query_generator.generate_query(),
            con = self.connection.endpoint,
            parse_dates = config.time_stamp
        )
        print(self.df.columns)
        
    def add_trig(self):
        days_per_month = 365/12
               
        self.df['sin_day'] = self.df['ts'].apply(
            lambda m : np.sin(m.hour*np.pi/12)
            )
        self.df['cos_day'] = self.df['ts'].apply(
            lambda m : np.cos(m.hour*np.pi/12)
            )
        self.df['sin_year'] = self.df['ts'].apply(
            lambda M : np.sin((M.day+M.month*days_per_month)*np.pi/365)
            )
        self.df['cos_year'] = self.df['ts'].apply(
            lambda M : np.cos((M.day+M.month*days_per_month)*np.pi/365)
            )

    def preprocess(self, method=None):
        if method == 'mean':    
            normalized_df=(self.df.drop(['ts'],axis=1)-self.df.drop(['ts'],axis=1).mean())/self.df.drop(['ts'],axis=1).std()
            normalized_df['ts'] = self.df['ts']  
            self.df = normalized_df
        elif method == 'min-max':
            normalized_df=(self.df.drop(['ts'],axis=1)-self.df.drop(['ts'],axis=1).min())/(self.df.drop(['ts'],axis=1).max()-self.df.drop(['ts'],axis=1).min())
            normalized_df['ts'] = self.df['ts']
            self.df = normalized_df
        else: 
            print('No preprocessing scheme specified')
            
    def plot_normalized(self):
        #df_std = (self.df.drop(['ts'],axis=1) - self.df.drop(['ts'],axis=1).mean()) / self.df.drop(['ts'],axis=1).std() 
        df_std = self.df.drop(['ts'],axis=1).melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        _ = ax.set_xticklabels(self.df.drop(['ts'],axis=1).keys(), rotation=90)
        plt.show()
        
            
    def important_frequencies(self,feature):
        fft = signal.rfft(self.df[feature])
        f_per_dataset = np.arange(0, len(fft))
        n_samples = len(df[feature]) # TODO
        
    def meta_data(self): # __repr__()?
        print(f"\nFeatures: {self.df.columns}\nNumber of samples: {len(self.df)}\nStart ts: {self.df['ts'].iloc[0]}\nEnd ts: {self.df['ts'].iloc[-1]}")
        
        
        

@dataclass
class DataSeries():
    '''
    This is a dataclass (which requires at leas Python 3.7)
    The attributes are only defined if they are available
    '''
    data : pd.DataFrame(columns = ['placeholder']) # Columns created by get_DataSeries method

    
    ''' REDUNDANT
    def add_date_to_df(self): # Work in progress
        self.data['date'] = [self.Date] * (self.data.index[-1]+1)
        self.data['date'] = pd.to_datetime(
            self.data['date'],
            dayfirst = False,
            format = config.dateformat)
        self.data['date'] = self.data['date'] + np.arange(0,self.data.index[-1]+1,dtype='timedelta64[s]')
        '''
            
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


