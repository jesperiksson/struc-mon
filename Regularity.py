import numpy as np
import scipy.signal as sps
import config

from SeriesObject import SeriesObject

class Regularity(SeriesObject):
    def __init__(self,r):
        super().__init__(r)
        
    def __repr__(self):
        return f"\nregularity duration: {self.duration/config.frequency:.3f}, regularity average: {self.abs_mu:.3f}"
