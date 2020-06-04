import numpy as np
import scipy as sp
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn import preprocessing

class DataBatch():
    def __init__(self, data, batch_num, speed, normalized_speed, category, damage_state):
        self.data = np.array(data)
        self.unnormalized_data = np.array(data)
        for i in range(np.shape(data)[0]):
            self.data[i,:] = self.data[i,:]/max(abs(self.data[i,:]))
        #self.data = preprocessing.normalize(self.data)
        self.sensors = np.shape(data)[0]
        self.batch_num = batch_num
        self.category = category
        self.n_steps = np.shape(self.data)[1]
        self.speed = {'km/h' : speed, 'm/s' : (speed*3.6/10)}
        self.normalized_speed = normalized_speed
        self.damage_state = damage_state
        self.normalized_damage_state = damage_state/100
        self.timestep = 0.001
        self.timesteps = np.arange(0, self.n_steps, 1)
        self.steps = [None]*self.sensors
        self.indices = [None]*self.sensors
        self.delta = [None]*self.sensors
        for i in range(self.sensors):
            self.steps[i] = self.n_steps

    def plot_batch(stack, arch, plot_sensor = '90'): 
        print(stack)
        side = min(int(np.floor(np.sqrt(len(stack)))),7)
        fig, axs = plt.subplots(side, side, constrained_layout=True)
        k = 0
        for i in range(side):            
            for j in range(side):        
                key = 'batch'+str(k%len(stack))
                axs[i][j].plot(stack[key].timesteps, stack[key].unnormalized_data[arch['sensors'][plot_sensor],:], 'b', linewidth=0.1)
                #plt.plot(stack[key].peaks_indices[sensor], stack[key].peaks[sensor], 'ro', linewidth = 0.4)
                axs[i][j].set_title(str(stack[key].speed[0]['km/h'])+'km/h')
                #plt.set_xlabel('timesteps')
                #plt.set_ylabel('acceleration')
                k += 1
            k += 1
        plt.suptitle(str(stack[key].damage_state)+'% Healthy at mid-span, registered at sensor: '+plot_sensor)
        name = 'E'+str(stack[key].damage_state)+'_d90_s'+plot_sensor+'.png'
        #print(name)
        #plt.savefig(name)
        plt.show()

        return

    def plot_series(self, plot_sensor = '90'):
        plt.plot(self.timesteps, self.data[sensors['sensors'][plot_sensor],:], 'b', linewidth=0.4)
        plt.plot(self.peaks_indices[sensor], self.peaks[sensors['sensors'][plot_sensor]], 'r*', mew = 0.02)
        plt.plot(self.extrema_indices[sensor], self.extrema[sensors['sensors'][plot_sensor]], 'go', mew = 0.02)
        plt.show()

class extrema(DataBatch):
    def __init__(self, data, batch_num, speed, normalized_speed, category = 'train', damage_state = 1):  
        super().__init__(data, batch_num, speed, normalized_speed)
        for i in range(self.sensors):
            indices = sp.signal.argrelextrema(
                np.absolute(self.data[i]), 
                np.greater, 
                axis = 0, 
                order = 1)
            self.indices[i] = indices[0]
            self.extrema[i] = self.data[i][self.indices[i]]
            self.steps[i] = np.shape(self.indices[i])[0]
            delta = np.diff(self.indices[i])
            self.delta[i] = delta/max(delta)

class peaks(DataBatch):
    def __init__(self, data, batch_num, speed, normalized_speed, category, damage_state):  
        super().__init__(data, batch_num, speed, normalized_speed, category, damage_state)
        self.peaks = [None]*self.sensors
        for i in range(self.sensors):
            self.indices[i], properties = sp.signal.find_peaks(
                self.data[i], 
                height = None, 
                threshold = None,
                distance = 2,
                prominence = None,
                width = None)
            self.peaks[i] = self.data[i][self.indices[i]]
            
            delta = np.diff(self.indices[i])
            self.delta[i] = delta/max(delta)
        self.n_steps = np.shape(self.peaks[0])[0] # overwrite data
        self.timesteps = self.indices[0]
        self.data = self.peaks 
            
class frequencySpectrum(DataBatch):
    def __init__(self, data, batch_num, speed, normalized_speed, category = 'train', damage_state = 1):  
        super().__init__(data, batch_num, speed, normalized_speed)#, category = 'train', damage_state = 1)           
        time = self.timestep*self.timesteps      #time vector
        Fs = 1/self.timestep                  #Samplig freq
        self.fourier = [None]*self.sensors
        self.frequency = [None]*self.sensors
        self.L = [None]*self.sensors
        for i in range(self.sensors): 
            self.fourier[i] = sp.fft(self.data[i])
            Tot_time = self.n_steps/Fs          #Total time for sample
            f_steps = 1/Tot_time                  #step between freq in plot
            self.frequency[i] = np.arange(0, int(Fs/f_steps))*f_steps
            S = 2.0/self.n_steps
            self.L[i] = S*abs(self.fourier[i])

    def plot_frequency(stack, sensors, plot_sensor = '90'):
        side = min(int(np.floor(np.sqrt(len(stack)))),4)
        fig, axs = plt.subplots(side, side, constrained_layout=True)      
        k = 0
        for i in range(side):            
            for j in range(side):        
                key = 'batch'+str(k%len(stack))
                axs[i][j].plot(stack[key].frequency[sensors['sensors'][plot_sensor]][:int(stack[key].n_steps/2)], stack[key].L[sensors['sensors'][plot_sensor]][:int(stack[key].n_steps/2)], 'b', linewidth=0.1)
                axs[i][j].set_title(str(stack[key].speed['km/h'])+'km/h')
                k += 1
            k += 1
        plt.suptitle('Spectrum plot '+str(stack[key].damage_state)+'% Healthy at mid-span, registered at sensor: '+str(plot_sensor))
        name = 'spectrum_E'+str(stack[key].damage_state)+'_d90_s'+str(plot_sensor)+'.png'
        plt.savefig(name)
        plt.show()
        



