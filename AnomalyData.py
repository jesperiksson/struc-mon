from Settings import Settings
import config

from Anomaly import Anomaly
from DataObjects import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from datetime import datetime, timedelta

class AnomalyData(DataObjects):
    def __init__(self,generator,connection):
        super().__init__(generator,connection)
        #self.anomalies = dict()
                
    def object_algorithm(self, df, foi, df_number = 0):
        end_of_patience = 0
        j = len(self.objects)
        anomaly_mode = False
        anomalies = dict()
        for i in range(len(df)):
            if abs(df[foi].iloc[i]) > self.object_settings.threshold: # Start or prolong anomaly_mode
                end_of_patience = i + self.object_settings.patience
                if anomaly_mode: # Means we were in anomaly mode previously
                    pass
                else: # We werent in anomaly_mode previously, reset counters
                    anomaly = []
                anomaly_mode = True # Regardless, either set or keep anomaly_mode True
            if anomaly_mode and i < end_of_patience: # We are in anomaly mode and it is not over
                anomaly.append(df[foi].iloc[i]) if abs(df[foi].iloc[i]) < self.object_settings.max_filter else anomaly.append(0)
            elif anomaly_mode and i == end_of_patience:
                anomaly.append(df[foi].iloc[i]) if abs(df[foi].iloc[i]) < self.object_settings.max_filter else anomaly.append(0)
                if len(anomaly) <= self.object_settings.patience + 1:
                    pass
                else: # Filter to remove anomalies with a single anomalous data point    
                    anomalies.update(
                        {j : Anomaly({
                            'array' : anomaly,
                            'end_index' : i,
                            'df_number' : df_number,
                            'feature' : foi,
                            'ts' : df['ts'].iloc[i]}
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
        
    def get_objects_name(self, name):
        return f"{config.anomaly_path}{name}.json"
