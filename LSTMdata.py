import numpy as np

class LSTMdata():

    def __init__(self, data, data_type = 'eval', n_batches = 1, train_percentage = 70, split_mode = 'last', pred_sensor = 0):
        self.data = data
        self.n_timesteps = np.shape(data)[1]
        self.n_series = np.shape(data)[0]
        self.data_type = data_type
        self.n_batches = n_batches
        self.n_sensors = int(self.n_series/self.n_batches)
        self.train_percentage = train_percentage
        self.split_mode = split_mode
        self.pred_sensor = pred_sensor
        self.n_train_batches = int(np.floor(self.train_percentage/100*self.n_batches))
        self.n_test_batches = self.n_batches - self.n_train_batches
        if split_mode == 'last':
            split_index = int(self.n_train_batches*self.n_sensors)
            if data_type == 'train':
                self.data = self.data[:split_index,:]
                # Patterns
                self.patterns = np.delete(self.data, pred_sensor, axis=1)
                reshaped_pattern = np.zeros((self.n_train_batches, self.n_sensors, self.n_timesteps-1))
                for i in range(self.n_train_batches-1):
                    reshaped_pattern[i,:,:] = self.patterns[i*self.n_sensors:(i+1)*self.n_sensors ,:]
                self.patterns = reshaped_pattern
                # Targets
                self.targets = np.reshape(self.data[:,pred_sensor],(self.n_train_batches, self.n_sensors))
                
            elif data_type == 'test':
                self.data = self.data[split_index:,:]
                # Patterns
                self.patterns = np.delete(self.data, pred_sensor, axis=1)
                reshaped_pattern = np.zeros((self.n_test_batches, self.n_sensors, self.n_timesteps-1))
                for i in range(self.n_train_batches-1):
                    reshaped_pattern[i,:,:] = self.patterns[i*self.n_sensors:(i+1)*self.n_sensors ,:]
                self.patterns = reshaped_pattern                
                # Targets
                self.targets = np.reshape(self.data[:,pred_sensor],(self.n_test_batches, self.n_sensors))
        elif split_mode == 'random':
            pass



