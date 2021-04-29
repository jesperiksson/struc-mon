import numpy as np
import scipy.signal as sps
import config
import ApEn
class Anomaly():
    def __init__(self, a):
        anomaly = np.array(a['array'])
        peaks, properties = sps.find_peaks(anomaly)
        self.anomaly = anomaly
        self.duration = len(anomaly)/config.frequency
        self.mu = anomaly.mean()
        self.abs_mu = abs(anomaly).mean()
        self.sigma = anomaly.var()
        self.max_a = abs(anomaly).max()
        self.rms = np.sqrt(np.mean(np.square(anomaly)))
        self.frequency = len(peaks)/len(anomaly)*config.frequency
        self.start_index = a['end_index'] - len(anomaly)
        self.end_index = a['end_index']
        self.df_number = a['df_number']
        self.feature = a['feature']
        self.ApEn = ApEn.ApEn(anomaly, 10, 3)
        #print(self.ApEn)
        
    def __repr__(self):
        return f"\nanomaly duration: {self.duration/config.frequency:.3f}, anomaly average: {self.abs_mu:.3f}"
        
    def get_feature_vector(self):
        return np.array([self.duration,self.mu,self.abs_mu,self.sigma,self.max_a,self.rms,self.frequency])#,self.ApEn])
        #return np.array([self.mu,self.abs_mu,self.sigma,self.max_a])
        
    def get_feature_labels(self):
        return ['Duration', 'mu', 'absolute_mu', 'sigma', 'max value','rms','frequency']#,'ApEn']
        #return ['mu', 'absolute mu', 'sigma', 'max value']
