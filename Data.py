# Standard packages
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import re
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# External packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose

#import tensorflow.signal as signal
#import rainflow as rf
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
        self.discontinuities = None
        self.dates = [self.query_generator.start_date]
        
    def find_discontinuities(self,tol = timedelta(hours=1)):     
        self.discontinuities = [i for i in range(len(self.df['ts'][:-1])) if self.df['ts'].iloc[i+1]-self.df['ts'].iloc[i]>tol]
        
    def split_at_discontinuities(self):
        self.dfs = []
        old = 0
        if self.discontinuities == None:
            self.dfs = [self.df]
        else: 
            for new in self.discontinuities:
                self.dfs.append(self.df[old:new])
                old=new+1
            self.dfs.append(self.df[old:-1])


        
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
        for i in range(len(self.dfs)):
            if method == 'mean': # standardize
                normalized_df=(self.dfs[i].drop(['ts'],axis=1)-self.dfs[i].drop(['ts'],axis=1).mean())/self.dfs[i].drop(['ts'],axis=1).std()
                normalized_df['ts'] = self.dfs[i]['ts']  
                self.dfs[i] = normalized_df
            elif method == 'min-max':
                normalized_df=(self.dfs[i].drop(['ts'],axis=1)-self.dfs[i].drop(['ts'],axis=1).min())/(self.dfs[i].drop(['ts'],axis=1).max()-self.dfs[i].drop(['ts'],axis=1).min())
                normalized_df['ts'] = self.dfs[i]['ts']
                self.dfs[i] = normalized_df
            else: 
                print('No preprocessing scheme specified')
                
    def distort(self,loc=0,scale=1):
        for i,df in enumerate(self.dfs):
            if i%2==0:
                ts = df['ts']
                df_size = df.drop(['ts'],axis=1).shape
                noise = pd.DataFrame(
                    np.random.normal(loc=loc,scale=scale,size=df_size),
                    columns = df.drop(['ts'],axis=1).columns,
                    index = df.index
                    )
                distorted = df.drop(['ts'],axis=1).add(noise)
                distorted['ts'] = ts
                distorted['distorted']=1
                self.dfs[i]=distorted
            else:              
                df['distorted']=0
                self.dfs[i]=df
            
            
    def plot_normalized(self):
        df_std = self.df.drop(['ts'],axis=1).melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        _ = ax.set_xticklabels(self.df.drop(['ts'],axis=1).keys(), rotation=90)
        plt.show()
        
            
    def fast_fourier_transform(self):
        for i in range(len(self.dfs)):
            ts = self.dfs[i]['ts']
            print(self.dfs[i].drop(['ts'],axis=1),self.dfs[i].columns.drop(['ts']))
            print(sp.fft.rfft(self.dfs[i].drop(['ts'],axis=1),axis=1,overwrite_x = True).shape)
            self.dfs[i] = pd.DataFrame(
                sp.fft.rfft(
                    self.dfs[i].drop(['ts'],axis=1),
                    axis=1,
                    overwrite_x = True),
                index=self.dfs[i].index,
                columns=self.dfs[i].columns.drop(['ts'])
                )
            self.dfs[i]['ts'] = ts
            
    def butter(self):
        for i in range(len(self.dfs)):
            ts = self.dfs[i]['ts']
            self.dfs[i] = pd.DataFrame(
                sp.signal.butter(
                    self.dfs[i].drop(['ts'],axis=1),
                    axis = 1,
                    overwrite_x = True),
                index=self.dfs[i].index,
                columns=self.dfs[i].columns.drop(['ts'])
                )
            self.dfs[i]['ts'] = ts
            
    def wawelet(self):
        for i in range(len(self.dfs)):
            ts = self.dfs[i]['ts']
            self.dfs[i] = pd.DataFrame(
                signal.cwt(
                    data = self.dfs[i].drop(
                        ['ts'],
                        axis=1),
                    wavelet = signal.ricker,
                    widths = np.arange(1, 31)*len(ts)),
                index=self.dfs[i].index,
                columns=self.dfs[i].columns.drop(['ts'])
                )
            self.dfs[i]['ts'] = ts      
            
    def ssa(self): 
        for i in range(len(self.dfs)): 
            pass
                
        
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
            #plt.show()
            
    def STL(self):
        for df in self.dfs:
            fig, axs = plt.subplots(len(df.columns.drop(['ts'])), figsize = config.figsize)
            for i,col in enumerate(df.columns.drop(['ts'])):
                result = seasonal_decompose(df[col], model='multiplicative')
                result.plot(ax = axs[i])
            plt.show()
                
        
        
    def generate_metadata_report(self,generator):
        ts_df = pd.read_sql_query(
            sql = self.query_generator.generate_metadata(),
            con = self.connection.endpoint,
            parse_dates = config.time_stamp,
            index_col = config.time_stamp
        )
        print(generator.generate_metadata_report(ts_df))
        
    def save_df(self,date):
        with open(f"{config.dataframe_path}{date}_df.json",'wb') as f:
            pickle.dump(self.df, f)
            
    def load_df(self,date):
        self.df = pickle.load(
            open(f"{config.dataframe_path}{date}_df.json",'rb')
            )
            
    def save_dfs(self,date):
        with open(f"{config.dataframes_path}{date}_dfs.json",'wb') as f:
            pickle.dump(self.dfs, f)
            
    def load_dfs(self,date):
        self.dfs = pickle.load(
            open(f"{config.dataframes_path}{date}_dfs.json",'rb')
            )
        self.dates = [date]
            
    def load_extend_dfs(self,date):
        self.dfs.extend(pickle.load(
            open(f"{config.dataframes_path}{date}_dfs.json",'rb')
            ))
        self.dates.extend([date])
        
        
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


