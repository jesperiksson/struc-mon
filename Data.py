import numpy as np
import pandas as pd
from dataclasses import dataclass
import os

from set_settings import set_settings
import config

@dataclass
class DataBatch():
    '''
    This is a dataclass (which requires at leas Python 3.7)
    The attributes are only defined if they are available
    '''
    data : pd.DataFrame(columns = ['placeholder'])
    #incl1 : np.ndarray = np.empty(0)
    #incl2 : np.ndarray = np.empty(0)
    #strain1 : np.ndarray = np.empty(0)
    #strain2 : np.ndarray = np.empty(0)
    
    MacId : int = 0
    NetworkId : int = 0
    DataAcqusitionCycle : int = 0
    DataAcqusitionDuration : int = 0
    SamplingRate : int = 0
    CutOffFrequency : int = 0
    Date : int = 0
        
    def normalize(self, scheme): # TBI, copied from Struc-mon 1
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
        
class Series_Stack(): 
    
    def __init__(self, settings, new_or_old):
        '''
        Goes to the location specified by 'file_path' in config.py and set these files as available.
        If the model is reloaded it goes to settings to see which files it already knows.
        The files that are available but not learned are set to be learned.
        '''
        if new_or_old == 'new':
            self.learned = set()
        elif new_or_old == 'old':
            self.learned = settings['learned']
        self.available = set()
        months = os.listdir(config.measurements)
        for i in range(len(months)):
            for j in range(len(config.sensors_of_interest)):
                try:
                    path = config.measurements + '/' + months[i] +'/'+ config.sensors_of_interest[j]
                    files = os.listdir(path)
                    self.available.update(set(path +'/'+ f for f in files))
                except FileNotFoundError: 
                    pass
        self.to_learn = list(self.available - self.learned)
        self.in_stack = set()
        self.settings = set_settings()
        self.stack = list()

    def populate_stack(self):
        '''
        Goes to all the 'to_learn' files and records them into DataBatch object with a pd.DataFrame
        '''
        for i in range(len(self.to_learn)):
            acc = pd.read_table(
                filepath_or_buffer = self.to_learn[i],
                delimiter = ';',
                header = 22, # The header row happens to be on line 22
                names = ['x', 'y', 'z','Index'])
            df = pd.DataFrame(acc)
            df['Index'] = df.index
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1] # move Index to front
            df = df[cols]
            content = self.read_file(self.to_learn[i])
            self.stack.append(self.get_Databatch(df, content))
            self.in_stack.update(self.to_learn[i])   
            
    def get_Databatch(self,df, aux_data):
        '''
        Records the meta data such as sampling rate, which sensor, det etc.
        '''
        data = df
        tstart = aux_data.find('Mac Id : ')+9
        MI = aux_data[tstart:tstart+16]
        tstart = aux_data.find('Network Id : ')+13
        NI = aux_data[tstart:tstart+4]
        tstart = aux_data.find('Data acquisition cycle : ')+25
        DAC = aux_data[tstart:tstart+5]
        tstart = aux_data.find('Data acquisition duration : ')+28
        DAD = aux_data[tstart:tstart+3]
        tstart = aux_data.find('Sampling rate : ')+15
        SR = aux_data[tstart:tstart+2]
        tstart = aux_data.find('Cut off frequency : ')+20
        COF = aux_data[tstart:tstart+2]
        tstart = aux_data.find('Date : ')+7
        DATE = aux_data[tstart:tstart+19]
        data = DataBatch(data,MI,NI,DAC,DAD,SR,COF,DATE)
        return data    
    

    
    def read_file(self,path): 
        ''' 
        Used by populate_stack()
        '''
        with open(path, 'r') as f:
            return f.read()
        
        
'''
    def read_file(path): # Used by populate_stack() DEPRECATED
        with open(path, 'r') as f:
        f = open(path, 'r')
        content = f.read()
        f.close()
        return content 
'''
        
'''
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
                axs[i][j].plot(
                    stack[key].frequency[sensors['sensors'][plot_sensor]][:int(stack[key].n_steps/2)], 
                    stack[key].L[sensors['sensors'][plot_sensor]][:int(stack[key].n_steps/2)], 
                    'b', 
                    linewidth=0.1)
                axs[i][j].set_title(str(stack[key].speed['km/h'])+'km/h')
                k += 1
            k += 1
        plt.suptitle(
            'Spectrum plot '+
            str(stack[key].damage_state)+
            '% Healthy at mid-span, registered at sensor: '
            +str(plot_sensor))
        name = 'spectrum_E'+str(stack[key].damage_state)+'_d90_s'+str(plot_sensor)+'.png'
        plt.savefig(name)
        plt.show()'''      



