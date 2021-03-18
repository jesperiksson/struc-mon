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
            
    def add_temp(self):
        temp_df = pd.read_sql_query(
            sql = self.query_generator.generate_temp_query(),
            con = self.connection.endpoint
        )
        increment = len(self.df)/len(temp_df)
        temp_col = np.empty(shape=len(self.df))
        temp_col[:] = np.nan
        for i in range(len(temp_df)):
            temp_col[int(i*increment)] = temp_df['temp'].iloc[i]
        self.df['temp'] = temp_col
        self.df.interpolate(
            method = 'linear',
            inplace = True)
            

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
        
    def train_test_split(self,data_split): # makes a dataframe out of all the smaller dataframes    
        n = len(self.df)
        self.train_df = self.df[0:int(n*data_split.train)]
        self.val_df = self.df[
            int(n*data_split.train):int(n*data_split.validation) + int(n*data_split.train)
            ]
        self.test_df = self.df[-int(n*data_split.test):]
        print(self.test_df)
        
    def plot_data(self):
        self.df.plot(
            y = self.df.columns.drop(['ts']),
            kind = 'line',
            grid = True,
            linewidth = 0.1
        )
        plt.show()
        
        
class NewData(Data):
    def __init__(self,query_generator,connection):
        super().__init__(query_generator,connection)
        
    def figure_out_length(self,model): # The amount of tuples needed for exactly one batch of data
        self.steps = model.time_series.total_window_size * model.settings_test.batch_size +1
        
    def make_new_df_postgres(self):
        self.df = pd.read_sql_query(
            sql = self.query_generator.generate_latest_query(steps = self.steps),
            con = self.connection.endpoint,
            parse_dates = config.time_stamp
        )


