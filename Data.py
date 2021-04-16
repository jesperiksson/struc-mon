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
from statsmodels.tsa.seasonal import seasonal_decompose
from pyts.decomposition import SingularSpectrumAnalysis

#import tensorflow.signal as signal
#import rainflow as rf
import seaborn as sns



# Self made modules
from Settings import Settings
import config
from SsaSettings import SsaSettings
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
            
    def add_temp(self):
        temp_df = pd.read_sql_query(
            sql = self.generator.generate_temp_query(),
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
        self.df = pickle.load(
            open(f"{config.dataframe_path}_{date}_df.json",'rb')
            )
            
    def save_dfs(self,name):
        with open(self.get_dfs_name(name),'wb') as f:
            pickle.dump(self.dfs, f)
            
    def load_dfs(self,date):
        self.dfs = pickle.load(
            open(self.get_dfs_name(date),'rb')
            )
        self.dates = [date]
            
    def load_extend_dfs(self,date):
        self.dfs.extend(pickle.load(
            open(self.get_dfs_name(name),'rb')
            ))
        self.dates.extend([date])
        
    def get_dfs_name(self,name):
        return f"{config.dataframes_path}_{name}_acc1_incl_dfs.json"
   
class PostgresData(Data):     
    def __init__(self,generator,connection):
        super().__init__(generator,connection)
        
    def make_df(self):
        self.df = pd.read_sql_query(
            sql = self.generator.generate_query(),
            con = self.connection.endpoint,
            parse_dates = config.time_stamp
        )
        self.discontinuities = None
        self.dates = [self.generator.start_date]
        
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
            
    def preprocess(self, method = 'mean'):
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
            
    def filter_hours(self, start_hour, end_hour):
        for i, df in enumerate(self.dfs):
            t1 = df['ts'].searchsorted(datetime.strptime(start_hour,config.dateformat_hms))
            t2 = df['ts'].searchsorted(datetime.strptime(end_hour,config.dateformat_hms))
            self.dfs[i] = df['ts'].loc[t1:t2-1]
            print(df['ts'].loc[t1:t2-1])
        
                
class NewData(PostgresData):

    def __init__(self,generator,connection):
        super().__init__(generator,connection)
        
    def figure_out_length(self,model): # The amount of tuples needed for exactly one batch of data
        #self.steps = model.time_series.total_window_size * model.settings_test.batch_size +1
        self.steps = (model.settings_model.input_time_steps * model.settings_model.shift) * model.settings_test.batch_size +1
        
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

class SSAData(Data):
    def __init__(self,generator,connection):
        super().__init__(generator,connection)
        
    def make_df(self):
        self.df = pd.read_sql_query(
            sql = self.generator.generate_query(),
            con = self.connection.endpoint,
            parse_dates = config.time_stamp
        )
        self.discontinuities = None
        self.dates = [self.generator.start_date]

    def ssa(self): 
        ssa_dfs = []
        self.ssa_settings = SsaSettings()
        
        for i, df in enumerate(self.dfs):
            df = df.drop(['ts'],axis=1)
            ssa_df = pd.DataFrame()
            n_samples = self.ssa_settings.get_n_samples(df)
            for j,col in enumerate(df.columns):
                X = df[col].iloc[:n_samples * self.ssa_settings.n_timestamps].to_numpy().reshape(
                    n_samples, self.ssa_settings.n_timestamps)
                ssa = SingularSpectrumAnalysis(
                    window_size = self.ssa_settings.window_size, 
                    groups = self.ssa_settings.groups)
                X_ssa = ssa.fit_transform(X)
                X_ssa_reshaped = X_ssa.reshape(self.ssa_settings.n_groups,-1)
                SSAs = [f"{col}_SSA{x+1}" for x in range(self.ssa_settings.n_groups)]
                for k, SSA in enumerate(SSAs):
                    ssa_df[SSA] = X_ssa_reshaped[k,:]
            ssa_dfs.append(ssa_df)
        self.dfs = ssa_dfs
        
class AggregatedData(Data):
    def __init__(self,generator,connection):
        super().__init__(generator,connection)
        
    def make_df(self):
        link = self.generator.generate_JSON_link()
        doi = ['avg','max','min','stddev']
        all_data_df = pd.DataFrame(columns = doi+['ts'])
        data, nextlink = self.get_dict_from_JSON(link)
        all_data_df = all_data_df.append(self.get_df(data),ignore_index=True)
        while nextlink is not None:
            data, nextlink = self.get_dict_from_JSON(nextlink)
            all_data_df = all_data_df.append(self.get_df(data),ignore_index=True)
        #all_data_df.set_index('ts', inplace=True)      
        self.df = all_data_df
        self.dfs = [self.df]
            
    
    def get_dict_from_JSON(self,link):
        response = urllib.request.urlopen(link)
        data = json.loads(response.read())
        try:
            nextlink = data['@iot.nextLink']
        except KeyError: 
            nextlink = None
        return data, nextlink
        
    def get_df(self,data):
        cols = config.doi+['ts']
        df = pd.DataFrame(columns = cols)
        for i in range(len(data['value'])):
            ts = datetime.strptime(data['value'][i]['resultTime'],config.dateformat_ymdhms)
            data_points = [data['value'][i]['result'][dp] for dp in config.doi]
            df=df.append(pd.DataFrame([data_points+[ts]],columns=cols),ignore_index=True)
        return df
        
    def plot_df(self,y='max',xbase=1000.,ybase=40.):
        fig, ax = plt.subplots()
        ax.plot(self.df.index,self.df[y],linewidth=0.1)
        xloc = plticker.MultipleLocator(base=xbase) # this locator puts ticks at regular intervals
        yloc = plticker.MultipleLocator(base=ybase)
        ax.xaxis.set_major_locator(xloc)
        ax.yaxis.set_major_locator(yloc)
        plt.show()
        
    def preprocess(self, method = 'mean'):
        for i in range(len(self.dfs)):
            print(self.dfs[i].columns)
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
        
    

