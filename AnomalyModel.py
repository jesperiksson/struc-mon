import numpy as np
from sklearn.preprocessing import normalize
class AnomalyModel():
    def __init__(self,anomaly_dict, settings):
        self.settings = settings
        self.X = np.transpose(np.array([x.get_feature_vector() for x in list(anomaly_dict.values())]))
        norm = normalize(self.X, norm = 'max', axis = 1, copy = False, return_norm = True)
