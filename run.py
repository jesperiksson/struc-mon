import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotli.pyplot as plt
from util import *

if __name__ == "__main__"

    foobar = readdata()
    LSTMdata = fit_to_LSTM(foobar) # Path to data file
    n_samples = 5
    print("\nStarting a LSTM machine")
    LSTM = LongShortTermMemoryMachine(LSTMdata, n_samples)
    LSTM.trainLSTM(LSTMdata) # To be continued
