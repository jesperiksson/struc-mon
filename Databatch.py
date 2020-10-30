import numpy as np
import pandas as pd
from dataclasses import dataclass
@dataclass
class DataBatch():
    data : pd.DataFrame(columns = ['placeholder'])
    #incl1 : np.ndarray = np.empty(0)
    #incl2 : np.ndarray = np.empty(0)
    #strain1 : np.ndarray = np.empty(0)
    #strain2 : np.ndarray = np.empty(0)
    
    MacId : int = 0
    MacId : int = 0
    NetworkId : int = 0
    DataAcqusitionCycle : int = 0
    DataAcqusitionDuration : int = 0
    SamplingRate : int = 0
    CutOffFrequency : int = 0
    Date : int = 0
        
    def normalize(self, scheme): # TBI
        self.l1 = preprocessing.normalize(
            X = self.raw_data,
            norm = 'l1')
        self.l2 = preprocessing.normalize(
            X = self.raw_data,
            norm = 'l2')
        self.max = preprocessing.normalize(
            X = self.raw_data,
            norm = 'max')
        self.data_dict = {
            'Raw data'  : self.raw_data,    # Unaltered
            'L-1'       : self.l1,          # L1-norm
            'L-2'       : self.l2,          # L2-norm
            'Maximum'   : self.max          # Normalized to the greatest acceleration
        }

            
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
        



