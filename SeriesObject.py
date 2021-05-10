import numpy as np
import scipy.signal as sps
import config
from datetime import datetime, timedelta
import minuter_med_buss as mmb

class SeriesObject():
    def __init__(self, r):
        series = np.array(r['array'])
        peaks, properties = sps.find_peaks(series)
        self.series = series
        self.duration = len(series)/config.frequency
        self.mu = series.mean()
        self.abs_mu = abs(series).mean()
        self.sigma = series.var()
        self.max_a = abs(series).max()
        self.rms = np.sqrt(np.mean(np.square(series)))
        self.frequency = len(peaks)/len(series)*config.frequency
        self.start_index = r['end_index'] - len(series)
        self.end_index = r['end_index']
        self.df_number = r['df_number']
        self.feature = r['feature']
        self.ts = r['ts']
        strp = self.ts
        if strp.date() != datetime(year=2021,month=5,day=4).date():
            self.irl_label = 0
        else:
            corr = strp - timedelta(seconds = 52)
            if corr.strftime('%Y-%m-%d %H:%M') in mmb.m:
                self.irl_label = 1
            else:
                self.irl_label = 0
        
    def get_feature_vector(self):
        #return np.array([self.duration,self.mu,self.abs_mu,self.sigma,self.max_a,self.rms,self.frequency])
        return np.array([self.mu,self.abs_mu,self.sigma,self.max_a,self.rms,self.frequency])
        
    def get_feature_labels(self):
        #return ['Duration', 'mu', 'absolute_mu', 'sigma', 'max value','rms','frequency']#,'ApEn']
        return ['mu', 'absolute_mu', 'sigma', 'max_value','rms','frequency']
        
    def get_feature_vector_sensor(self): # The one used for PCA
        return np.array(
            [self.duration,self.mu,self.abs_mu,self.sigma,self.max_a,
            self.rms,self.frequency])
            
    def get_feature_labels_sensor(self):
        return ['Duration', 'mu', 'absolute_mu', 'sigma', 'max_value','rms','frequency','sensor']
