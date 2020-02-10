import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf import keras as ks

''' Utilities for various classes'''

def fit_to_LSTM(data):
    """
    Function that fits raw data into a format that fits LSTM, i.e. array of arrays of acceleration signals in time-domain
    """
    LSTMdata = np.zeros(data.shape) # placeholder
    return LSTMdata
