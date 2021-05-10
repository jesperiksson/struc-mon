from Settings import Settings
import config
from DataObjects import DataObjects

from Regularity import Regularity
from DataObjects import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from datetime import datetime, timedelta

class RegularityData(DataObjects):
    def __init__(self,generator,connection):
        super().__init__(generator,connection)
        #self.regularities = dict()
        
    def object_algorithm(self, df, foi, df_number = 0):
        end_of_patience = 0
        j = len(self.objects)
        anomaly_mode = False
        regularity = []
        regularities = dict()
        for i in range(len(df)):   
            if abs(df[foi].iloc[i]) > self.object_settings.threshold: # Start or prolong anomaly_mode
                end_of_patience = i + self.object_settings.patience
                anomaly_mode = True # Regardless, either set or keep anomaly_mode True  
                
            if anomaly_mode and i + self.object_settings.patience == end_of_patience: # Anomaly just started
                try:
                    regularities.update(
                        {j : Regularity({
                            'array' : regularity,
                            'end_index' : i-1,
                            'df_number' : df_number,
                            'feature' : foi}
                            )})
                    j+=1
                except ValueError:
                    #print('empty')
                    pass
            elif anomaly_mode and i < end_of_patience:
                pass
            elif anomaly_mode and i == end_of_patience:
                regularity = []
                anomaly_mode = False
            else:
                regularity.append(df[foi].iloc[i]) if abs(df[foi].iloc[i]) < self.object_settings.max_filter else anomaly.append(0)                         
        return regularities
    
    def plot_regularities(self, df_num, regularity_key, start = 0, end = -1):
        plt.plot(
            self.dfs[df_num]['ts'][start:end],
            self.dfs[df_num][self.regularity[regularity_key].feature][start:end],
            linestyle = 'solid',
            linewidth = 0.1,
            color = 'tab:blue')
        plt.grid(axis = 'y', color = '0.95')
        start_index = self.objects[regularity_key].end_index-self.objects[regularity_key].duration
        end_index = self.objects[regularity_key].end_index
        plt.plot(
            self.dfs[df_num]['ts'][start_index:end_index],
            self.objects[regularity_key].regularity,
            linestyle = 'solid',
            linewidth = 0.15,
            color = 'tab:orange')
        plt.show()
        
    def get_objects_name(self, name):
        return f"{config.regularity_path}{name}.json"
