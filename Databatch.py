import numpy as np
import scipy as sp
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn import preprocessing

#np.set_printoptions(threshold=sys.maxsize)

class DataBatch():
    def __init__(self, data, batch_num, speed, normalized_speed, element, category = 'train', damage_state = 1):
        self.data = np.array(data)
        for i in range(np.shape(data)[0]):
            self.data[i,:] = self.data[i,:]/max(abs(self.data[i,:]))
        #self.data = preprocessing.normalize(self.data)
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
        self.indices = [None]*sensors
        self.steps = [None]*sensors
        self.delta = [None]*sensors
        method = 'extrema'
        if method == 'extrema':
            self.peaks = [None]*sensors
            for i in range(sensors):
                indices = sp.signal.argrelextrema(
                    np.absolute(self.data[i]), 
                    np.greater, 
                    axis = 0, 
                    order = 1)
                self.indices[i] = indices[0]
                self.peaks[i] = self.data[i][self.indices[i]]
                self.steps[i] = np.shape(self.indices[i])[0]
                delta = np.diff(self.indices[i])
                self.delta[i] = delta/max(delta)
        elif method == 'peaks':
            self.peaks = [None]*sensors
            for i in range(sensors):
                self.indices[i], properties = sp.signal.find_peaks(
                    self.data[i], 
                    height = None, 
                    threshold = None,
                    distance = 2,
                    prominence = None,
                    width = None)
                self.peaks[i] = self.data[i][self.indices[i]]
                self.steps[i] = np.shape(self.peaks[i])[0]
                delta = np.diff(self.indices[i])
                self.delta[i] = delta/max(delta)
                self.delta[i].extend(0)
        #print(self.peaks_delta)
            
   
    ### SPEKTRUM ANALYS ###
        X=sp.fftpack.fft(self.data)

        velocity=speed*3.6/10               
        t=self.timestep*self.timesteps      #time vector
        Fs=1/self.timestep                  #Samplig freq
        
        Tot_time=len(self.data)/Fs          #Total time for sample
        f_steps=1/Tot_time                  #step between freq in plot

        self.f=np.array(range(0, int(Fs/f_steg)))*f_steg

        S = 2.0/len(t.steps)
        self.L = S*abs(X)
    ### End ###
    def plot_batch(stack, sensor = '1/2'):
        sensor_dict = {
            '1/18' : 0,
            '1/4'  : 1,
            '1/2'  : 2,
            '3/4'  : 3,
            '17/18': 4}
        sensor = sensor_dict['17/18']
        side = min(int(np.floor(np.sqrt(len(stack)))),7)
        fig, axs = plt.subplots(side, side, constrained_layout=True)
        k = 0
        for i in range(side):            
            for j in range(side):        
                key = 'batch'+str(k%len(stack))
                axs[i][j].plot(stack[key].timesteps, stack[key].data[sensor,:], 'b', linewidth=0.1)
                #plt.plot(stack[key].peaks_indices[sensor], stack[key].peaks[sensor], 'ro', linewidth = 0.4)
                axs[i][j].set_title(str(stack[key].speed)+'km/h')
                #plt.set_xlabel('timesteps')
                #plt.set_ylabel('acceleration')
                k += 1
            k += 1
        plt.suptitle(str(stack[key].damage_state)+'% Healthy at mid-span, registered at sensor: '+str(sensor+1))
        name = 'E'+str(stack[key].damage_state)+'_d90_s'+str(sensor)+'.png'
        #print(name)
        plt.savefig(name)
        #plt.show()

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
        plt.plot(self.peaks_indices[sensor], self.peaks[sensor], 'r*', mew = 0.02)
        plt.plot(self.extrema_indices[sensor], self.extrema[sensor], 'go', mew = 0.02)
        plt.show()



