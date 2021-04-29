from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import pandas as pd

class KMeansClustering():
    def __init__(self,anomaly_dict, settings):
        self.settings = settings
        self.X = np.transpose(np.array([x.get_feature_vector() for x in list(anomaly_dict.values())]))
        norm = normalize(self.X, norm = 'max', axis = 1, copy = False, return_norm = True)
        self.kmeans = KMeans(
            n_clusters = settings.n_clusters,
            init = settings.init,
            n_init = settings.n_init,
            max_iter = settings.max_iter,
            tol = settings.tol,
            verbose = settings.verbose)
#            algortihm = 'full')
            
    def fit_Kmeans(self):
        self.kmeans.fit_transform(self.X.transpose())
        print(self.kmeans.labels_)
        
    def send_labels(self):
        return self.kmeans.labels_
