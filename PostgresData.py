from Settings import Settings
import config

from Anomaly import Anomaly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from datetime import datetime, timedelta
import random

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
            
class AnomalyData(PostgresData):
    def __init__(self,generator,connection):
        super().__init__(generator,connection)
        self.anomalies = dict()
        self.anomalies_dict = dict()
        
    def filter_hours(self, start_hour, end_hour):
        self.time_filtered_dfs = []
        for i, df in enumerate(self.dfs):
            try:
                date = datetime.strftime(df['ts'].iloc[0].date(),config.dateformat)
                t1 = df['ts'].searchsorted(datetime.strptime(f"{date} {start_hour}",config.dateformat_ymdhms))
                t2 = df['ts'].searchsorted(datetime.strptime(f"{date} {end_hour}",config.dateformat_ymdhms))
                #if t1 != 0 or t2 != 0:
                if t2 != 0:
                    self.time_filtered_dfs.append(df.loc[t1:t2-1])
                    #print(df.loc[t1:t2-1])
            except IndexError:
                pass

    def plot_filtered_hours(self, foi = 'acc1_ch_z', plot_anomalies = True, project_anomalies = None):
        max_plots = 1
        n_plots = min(max_plots,len(self.dfs))
        fig, axs = plt.subplots(n_plots, figsize = config.figsize)
        randints = [random.randint(0,len(self.dfs)-1) for p in range(n_plots)]
        for i in range(n_plots):
            self.dfs[randints[i]].plot(
                x = 'ts', y = foi, kind = 'line', ax = axs[i], grid = True, linewidth = 0.1, sharey = True
                )
            try:
                axs[i].set_title(f"{datetime.strftime(self.dfs[randints[i]]['ts'].iloc[0],config.dateformat)}")
            except IndexError:
                axs[i].set_title(f"")
            if plot_anomalies:
                anomalies = self.anomaly_algorithm(self.dfs[randints[i]],foi)
                for key in anomalies.keys():
                    axs[i].plot(
                        self.dfs[randints[i]]['ts'][anomalies[key].start_index+1: anomalies[key].end_index+1],
                        anomalies[key].anomaly,
                        linestyle = 'solid',
                        linewidth = 0.15,
                        color = 'tab:orange')
                    axs[i].grid(axis = 'y', color = '0.95')
            if project_anomalies in self.dfs[randints[i]].columns:
                anomalies = self.anomaly_algorithm(self.dfs[randints[i]],project_anomalies)
                for key in anomalies.keys():
                    axs[i].plot(
                        self.dfs[randints[i]]['ts'][anomalies[key].start_index+1: anomalies[key].end_index+1],
                        self.dfs[randints[i]][project_anomalies][anomalies[key].start_index+1: anomalies[key].end_index+1],
                        linestyle = 'solid',
                        linewidth = 0.15,
                        color = 'tab:green')
                    axs[i].grid(axis = 'y', color = '0.95')                
        fig.tight_layout()    
        plt.show()
        
    def save_plots(self, foi = 'acc1_ch_z', plot_anomalies = True):
        for i, df in enumerate(self.time_filtered_dfs):
            if len(df) > 2:
                df.plot(
                    x = 'ts', y = foi, kind = 'line', grid = True, linewidth = 0.1, #sharex = True
                    )
                try:
                    date_of_day = f"{datetime.strftime(df['ts'].iloc[0],config.dateformat)}"
                    plt.title(date_of_day)
                except IndexError:
                    plt.title(f"")
                if plot_anomalies:
                    anomalies = self.anomaly_algorithm(df,foi)
                    for key in anomalies.keys():
                        plt.plot(
                            df['ts'][anomalies[key].start_index+1: anomalies[key].end_index+1],
                            anomalies[key].anomaly,
                            linestyle = 'solid',
                            linewidth = 0.15,
                            color = 'tab:orange')
                        plt.grid(axis = 'y', color = '0.95')
                fname = f"{config.fig_path}{date_of_day}{df['ts'][df.first_valid_index()]}.png"
                plt.savefig(
                    fname = fname,
                    dpi = 'figure',) 
                plt.close()          
        
        
    def merge_dfs(self): # Recreate df from dfs
        self.df = pd.DataFrame(columns=self.dfs[0].columns)
        for i,df in enumerate(self.dfs):
            self.df = self.df.append(df,ignore_index=True)
        
    def set_anomaly_settings(self,anomaly_settings):
        self.anomaly_settings = anomaly_settings 

    def locate_anomalies(self,feature):        
        anomalies = self.anomaly_algorithm(self.df,feature)
        self.anomalies.update(anomalies)
        
    def locate_anomalies_dfs(self,feature):
        for i,df in enumerate(self.dfs):
            a = self.anomaly_algorithm(df,feature,i)
            self.anomalies.update(a)
            self.anomalies_dict.update({feature:a})
            
    def locate_anomalies_filtered_dfs(self,feature):
        for i,df in enumerate(self.time_filtered_dfs):
            a = self.anomaly_algorithm(df,feature,i)
            self.anomalies.update(a)
            self.anomalies_dict.update({feature:a})
        
    def anomaly_algorithm(self, df, foi, df_number = 0):
        end_of_patience = 0
        j = len(self.anomalies)
        anomaly_mode = False
        anomalies = dict()
        for i in range(len(df)):
            if abs(df[foi].iloc[i]) > self.anomaly_settings.threshold: # Start or prolong anomaly_mode
                end_of_patience = i + self.anomaly_settings.patience
                if anomaly_mode: # Means we were in anomaly mode previously
                    pass
                else: # We werent in anomaly_mode previously, reset counters
                    anomaly = []
                anomaly_mode = True # Regardless, either set or keep anomaly_mode True
            if anomaly_mode and i < end_of_patience: # We are in anomaly mode and it is not over
                anomaly.append(df[foi].iloc[i]) if abs(df[foi].iloc[i]) < self.anomaly_settings.max_filter else anomaly.append(0)
            elif anomaly_mode and i == end_of_patience:
                anomaly.append(df[foi].iloc[i]) if abs(df[foi].iloc[i]) < self.anomaly_settings.max_filter else anomaly.append(0)
                if len(anomaly) <= self.anomaly_settings.patience + 1:
                    pass
                else: # Filter to remove anomalies with a single anomalous data point    
                    anomalies.update(
                        {j : Anomaly({'array' : anomaly,'end_index' : i,'df_number' : df_number,'feature' : foi}
                            )})
                    j+=1
                anomaly_mode = False
            else:
                pass
        return anomalies    
    
    
    def plot_anomalies(self, df_num, anomaly_key, start = 0, end = -1):
        plt.plot(
            self.dfs[df_num]['ts'][start:end],
            self.dfs[df_num][self.anomalies[anomaly_key].feature][start:end],
            linestyle = 'solid',
            linewidth = 0.1,
            color = 'tab:blue')
        plt.grid(axis = 'y', color = '0.95')
        start_index = self.anomalies[anomaly_key].end_index-self.anomalies[anomaly_key].duration
        end_index = self.anomalies[anomaly_key].end_index
        plt.plot(
            self.dfs[df_num]['ts'][start_index:end_index],
            self.anomalies[anomaly_key].anomaly,
            linestyle = 'solid',
            linewidth = 0.15,
            color = 'tab:orange')
        plt.show()
