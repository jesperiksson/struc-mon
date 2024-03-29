import numpy as np
import scipy.signal as sps
import config

from SeriesObject import SeriesObject

class Anomaly(SeriesObject):
    def __init__(self,r):
        super().__init__(r)
        
    def __repr__(self):
        return f"\nanomaly duration: {self.duration/config.frequency:.3f}, anomaly average: {self.abs_mu:.3f}"
        

