import numpy as np
import scipy as sp
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from sklearn import preprocessing

#np.set_printoptions(threshold=sys.maxsize)

class DataBatch():
    def __init__(self, data, batch_num, speed, normalized_speed, element, category = 'train', damage_state = 1):
        self.data = np.array(data)
        self.data = preprocessing.normalize(self.data)
        sensors = np.shape(data)[0]
        self.batch_num = batch_num
        self.category = category
        self.n_steps = np.shape(self.data)[1]
        self.speed = speed
        self.normalized_speed = normalized_speed
        self.element = element/180
        self.damage_state = damage_state
        self.timestep = 0.001
        self.timesteps = np.arange(0, self.n_steps, 1)
        assert self.element <= 1
        self.batch = {'1/18' : np.array(data[0]),
                      '1/4'  : np.array(data[1]),
                      '1/2'  : np.array(data[2]),
                      '3/4'  : np.array(data[3]),
                      '17/18': np.array(data[4])
                      }
        self.extrema = [None]*sensors
        self.peaks = [None]*sensors
        self.extrema_indices = [None]*sensors
        self.peaks_indices = [None]*sensors
        for i in range(sensors):
            indices = sp.signal.argrelextrema(
                np.absolute(data[i]), 
                np.greater, 
                axis = 0, 
                order = 1)
            self.extrema_indices[i] = indices[0]
            self.extrema[i] = self.data[i][self.extrema_indices[i]]
        
            self.peaks_indices[i], properties = sp.signal.find_peaks(
                self.data[i], 
                height = None, 
                threshold = None,
                distance = 2,
                prominence = None,
                width = None)
            self.peaks[i] = self.data[i][self.peaks_indices[i]]
        self.peak_steps = np.shape(self.peaks[0])[0]

    def plot_batch(self, sensor = '1/2'):
        sensor_dict = {
            '1/18' : 0,
            '1/4'  : 1,
            '1/2'  : 2,
            '3/4'  : 3,
            '17/18': 4}
        fig, axs = plt.subplots(1, 1, constrained_layout = True, squeeze = False)
        print(axs)
        for i in range(len(sensor_list)):
            axs[i].plot(self.timesteps, self.data[sensor_dict[sensor_list[i]],:], 'b', linewidth=0.4)
            axs[i].plot(self.peaks[0,:], self.peaks[1,:], 'ro', linewidth = 0.4)
            axs[i].set_title(sensor_dict[sensor_list[i]])
            axs[i].set_xlabel('timesteps')
            axs[i].set_ylabel('acceleration')
        plt.suptitle('Batch '+str(self.batch_num))
        plt.show()
        return

    def plot_series(self, plot_sensor = '1/2'):
        sensor_dict = {
            '1/18' : 0,
            '1/4'  : 1,
            '1/2'  : 2,
            '3/4'  : 3,
            '17/18': 4}
        sensor = sensor_dict['1/2']
        plt.plot(self.timesteps, self.data[sensor,:], 'b', linewidth=0.4)
        print(np.shape(self.extrema_indices[2]), np.shape(self.extrema[2]))
        plt.plot(self.peaks_indices[sensor], self.peaks[sensor], 'r*', mew = 0.02)
        plt.plot(self.extrema_indices[sensor], self.extrema[sensor], 'go', mew = 0.02)
        plt.show()



