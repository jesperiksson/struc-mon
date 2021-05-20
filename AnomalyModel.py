'''
Parent class for models operating on anomaly objects
'''
import numpy as np
from sklearn.preprocessing import normalize
class AnomalyModel():
    def __init__(self,anomaly_dict, settings, features):
        self.settings = settings
        self.features = features
        self.X = np.transpose(np.array([[x.get_feature_dict()[key] for key in features] for x in list(anomaly_dict.values())]))
        norm = normalize(self.X, norm = 'max', axis = 1, copy = False, return_norm = True) # inplace normalization
        self.feature_labels = list(next(iter(anomaly_dict.values())).get_feature_dict().keys())
