class DataBatch():
    def __init__(self, a, data, batch_num, speed, normalized_speed, category, damage_state):
        self.data = np.array(data)
        self.raw_data = np.array(data)
        #for i in range(np.shape(data)[0]):
        #    self.data[i,:] = self.raw_data[i,:]/max(abs(self.raw_data[i,:]))
        self.l1 = preprocessing.normalize(
            X = self.raw_data,
            norm = 'l1')
        self.l2 = preprocessing.normalize(
            X = self.raw_data,
            norm = 'l2')
        self.max = preprocessing.normalize(
            X = self.raw_data,
            norm = 'max')
        self.sensors = np.shape(data)[0]
        self.batch_num = batch_num
        self.category = category
        self.n_steps = np.shape(self.data)[1]
        speed_dict = {
            'km/h' : speed, 
            'm/s' : (speed*3.6/10)
            }
        self.speed = speed_dict[a['speed_unit']]
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
        self.data_dict = {
            'Raw data'  : self.raw_data,    # Unaltered
            'L-1'       : self.l1,          # L1-norm
            'L-2'       : self.l2,          # L2-norm
            'Maximum'   : self.max          # Normalized to the greatest acceleration
        }
        #self.data = self.data_dict[a['normalization']]
        self.n_series = int(self.n_steps)-int(a['delta']*a['n_pattern_steps'])

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
        



