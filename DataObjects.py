'''
Child class to PostgresData. Parent class to the classes of AnomalyData and RegularityData 
'''
from Settings import Settings
import config

from Anomaly import Anomaly
from Regularity import Regularity
from PostgresData import PostgresData

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import random
import pickle

class DataObjects(PostgresData):
    def __init__(self,generator,connection):
        super().__init__(generator,connection)
        self.objects = dict()
        
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
                
    def purge_empty_time_filtered_dfs(self):
        self.time_filtered_dfs = [df if len(df)>1 else None for df in self.time_filtered_dfs]
        self.time_filtered_dfs = [i for i in self.time_filtered_dfs if isinstance(i,pd.DataFrame)]

    def plot_filtered_hours(self, foi = 'acc1_ch_z', plot_objects = True, project_objects = None):
        max_plots = 2
        n_plots = min(max_plots,len(self.time_filtered_dfs))
        fig, axs = plt.subplots(n_plots, figsize = config.figsize)
        randints = [random.randint(0,len(self.time_filtered_dfs)-1) for p in range(n_plots)]
        for i in range(n_plots):
            self.time_filtered_dfs[randints[i]].plot(
                x = 'ts', y = foi, kind = 'line', ax = axs[i], grid = True, linewidth = 0.1, sharey = True
                )
            fmt_minute = mdates.MinuteLocator()
            axs[i].xaxis.set_minor_locator(fmt_minute)
            axs[i].xaxis.grid(True, which = 'minor')
            try:
                axs[i].set_title(f"{datetime.strftime(self.dfs[randints[i]]['ts'].iloc[0],config.dateformat)}")
            except IndexError:
                axs[i].set_title(f"")
            if plot_objects:
                objects = self.object_algorithm(self.time_filtered_dfs[randints[i]],foi)
                for key in objects.keys():
                    axs[i].plot(
                        self.time_filtered_dfs[randints[i]]['ts'][objects[key].start_index+1: objects[key].end_index+1],
                        objects[key].series,
                        linestyle = 'solid',
                        linewidth = 0.15,
                        color = 'tab:orange')
                    axs[i].grid(axis = 'y', color = '0.95')
            if project_objects in self.time_filtered_dfs[randints[i]].columns:
                objects = self.object_algorithm(self.time_filtered_dfs[randints[i]],project_objects)
                for key in objects.keys():
                    axs[i].plot(
                        self.dfs[randints[i]]['ts'][objects[key].start_index+1: objects[key].end_index+1],
                        self.dfs[randints[i]][project_objects][objects[key].start_index+1: objects[key].end_index+1],
                        linestyle = 'solid',
                        linewidth = 0.15,
                        color = 'tab:green')
                    axs[i].grid(axis = 'y', color = '0.95')                
        fig.tight_layout()    
        plt.show()
        
    def save_plots(self, foi = 'acc1_ch_z', plot_objects = True):
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
                if plot_objects:
                    objects = self.object_algorithm(df,foi)
                    for key in objects.keys():
                        plt.plot(
                            df['ts'][objects[key].start_index+1: objects[key].end_index+1],
                            objects[key].object,
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
        
    def set_object_settings(self,object_settings):
        self.object_settings = object_settings 

    def locate_objects(self,feature):        
        objects = self.object_algorithm(self.df,feature)
        self.objects.update(objects)
        
    def save_objects(self,name):
        with open(self.get_objects_name(name),"wb") as f:
            pickle.dump(self.objects, f)
            
    def load_objects(self,name):
        self.objects = pickle.load(open(self.get_objects_name(name),'rb'))
        
    def locate_objects_dfs(self,feature):
        for i,df in enumerate(self.dfs):
            a = self.object_algorithm(df,feature,i)
            self.objects.update(a)
            
    def locate_objects_filtered_dfs(self,feature):
        for i,df in enumerate(self.time_filtered_dfs):
            a = self.object_algorithm(df,feature,i)
            self.objects.update(a)
        

