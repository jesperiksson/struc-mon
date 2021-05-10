from Settings import Settings
import config

from Anomaly import Anomaly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from datetime import datetime, timedelta
import random
import pickle

from Data import Data
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
        
    def purge_empty_dfs(self):
        self.dfs = [df if len(df)>1 else None for df in self.dfs]
        self.dfs = [i for i in self.dfs if isinstance(i,pd.DataFrame)]
        
    def plot_data(self):
        for df in self.dfs:
            fig, axs = plt.subplots(len(df.columns.drop(['ts'])), figsize = config.figsize)
            for i,col in enumerate(df.columns.drop(['ts'])):               
                df.plot(
                    y = col,
                    kind = 'line',
                    ax = axs[i],
                    grid = True,
                    linewidth = 0.1,
            )
            print(df['ts'])
            plt.show()
            
    def preprocess(self, method = 'standard'):
        for i in range(len(self.dfs)):
            if method == 'standard': # standardize
                normalized_df=(self.dfs[i].drop(['ts'],axis=1)-self.dfs[i].drop(['ts'],axis=1).mean())/self.dfs[i].drop(['ts'],axis=1).std()
                normalized_df['ts'] = self.dfs[i]['ts']  
                self.dfs[i] = normalized_df
            elif method == 'min-max': # 0 to 1
                normalized_df=(self.dfs[i].drop(['ts'],axis=1)-self.dfs[i].drop(['ts'],axis=1).min())/(self.dfs[i].drop(['ts'],axis=1).max()-self.dfs[i].drop(['ts'],axis=1).min())
                normalized_df['ts'] = self.dfs[i]['ts']
                self.dfs[i] = normalized_df
            else: 
                print('No preprocessing scheme specified')
                
    def find_correlation(self):
        print(self.df[self.df.drop(['ts'],axis=1).columns].corr())
                
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
            

