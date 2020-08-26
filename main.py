
# Imports
# Other files
import Menu
import functions
import models
import settings
import NeuralNet
import DataBatch

# Libraries and modules
import datetime
import time
import pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn import preprocessing

from tensorflow import keras
from tensorflow.python.keras.models import Sequential, Model, model_from_json
from tensorflow.python.keras.layers import Input, Dense, LSTM, CuDNNLSTM, concatenate, Activation, Reshape, Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import metrics, regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizers import RMSprop

def main():
    menu()

if __name__ == "__main__":
    main()
