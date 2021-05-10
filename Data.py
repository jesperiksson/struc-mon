# Standard packages
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import re
import pickle
from abc import ABC, abstractmethod
import urllib.request, json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# External packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import scipy as sp
from scipy import signal


#import tensorflow.signal as signal
#import rainflow as rf
import seaborn as sns



# Self made modules
from Settings import Settings
import config 
#import FilterSettings
#import RainflowSettings

class Data(ABC):
    def __init__(self,generator,connection):
        self.generator = generator
        self.connection = connection
     
    @abstractmethod    
    def make_df(self):
        pass
        
    @abstractmethod    
    def preprocess(self):
        pass
        
    def add_trig(self):
        for i in range(len(self.dfs)):
            days_per_month = 365/12
                   
            self.dfs[i]['sin_day'] = self.df['ts'].apply(
                lambda m : np.sin(m.hour*np.pi/12)
                )
            self.dfs[i]['cos_day'] = self.df['ts'].apply(
                lambda m : np.cos(m.hour*np.pi/12)
                )
            self.dfs[i]['sin_year'] = self.df['ts'].apply(
                lambda M : np.sin((M.day+M.month*days_per_month)*np.pi/365)
                )
            self.dfs[i]['cos_year'] = self.df['ts'].apply(
                lambda M : np.cos((M.day+M.month*days_per_month)*np.pi/365)
                )
            
            
    def plot_normalized(self):
        df_std = self.df.drop(['ts'],axis=1).melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        _ = ax.set_xticklabels(self.df.drop(['ts'],axis=1).keys(), rotation=90)
        plt.show()
                         
        
    def meta_data(self): # __repr__()?
        print(f"\nFeatures: {self.df.columns}\nNumber of samples: {len(self.df)}\nStart ts: {self.df['ts'].iloc[0]}\nEnd ts: {self.df['ts'].iloc[-1]}")
        
    def train_test_split(self,data_split): # makes a dataframe out of all the smaller dataframes    
        self.train_dfs = []
        self.val_dfs = []
        self.test_dfs = []
        for i in range(len(self.dfs)):       
            n = len(self.dfs[i])
            self.train_dfs.append(self.dfs[i][0:int(n*data_split.train)])
            self.val_dfs.append(self.dfs[i][int(n*data_split.train):int(n*data_split.validation) + int(n*data_split.train)])
            self.test_dfs.append(self.dfs[i][-int(n*data_split.test):])
        
    def plot_data(self):
        for df in self.dfs:
            fig, axs = plt.subplots(len(df.columns.drop(['ts'])), figsize = config.figsize)
            for i,col in enumerate(df.columns.drop(['ts'])):               
                df.plot(
                    #x = 'ts',
                    y = col,
                    kind = 'line',
                    ax = axs[i],
                    grid = True,
                    linewidth = 0.1,
                    #xticks = df['ts'][0:-1:1000],
                    #sharex = True
            )
            print(df['ts'])
            plt.show()
            
    def STL(self):
        for df in self.dfs:
            fig, axs = plt.subplots(len(df.columns.drop(['ts'])), figsize = config.figsize)
            for i,col in enumerate(df.columns.drop(['ts'])):
                result = seasonal_decompose(df[col], model='multiplicative')
                result.plot(ax = axs[i])
            plt.show()
                
        
        
    def generate_metadata_report(self,generator):
        ts_df = pd.read_sql_query(
            sql = self.generator.generate_metadata(),
            con = self.connection.endpoint,
            parse_dates = config.time_stamp,
            index_col = config.time_stamp
        )
        print(generator.generate_metadata_report(ts_df))
        
    def save_df(self,date):
        with open(f"{config.dataframe_path}_{date}_df.json",'wb') as f:
            pickle.dump(self.df, f)
            
    def load_df(self,date):
        self.df = pickle.load(open(f"{config.dataframe_path}_{date}_df.json",'rb'))
            
    def save_dfs(self,name):
        with open(self.get_dfs_name(name),'wb') as f:
            pickle.dump(self.dfs, f)
            
    def load_dfs(self,date):
        self.dfs = pickle.load(open(self.get_dfs_name(date),'rb'))
        self.dates = [date]
            
    def load_extend_dfs(self,date):
        self.dfs.extend(pickle.load(open(self.get_dfs_name(date),'rb')))
        self.dates.extend([date])
        
    def get_dfs_name(self,name):
        return f"{config.dataframes_path}_{name}_acc1_incl_dfs.json"
   
