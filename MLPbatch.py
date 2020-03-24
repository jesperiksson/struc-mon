import numpy as np

class MLPbatch():
    def __init__(self, data, batch_num, diff, category = 'train'):
        self.data = data
        self.batch_num = batch_num
        self.diff = diff
        self.category = category
        self.n_steps = np.shape(self.data)[1]
        self.n_sensors = np.shape(self.data)[0]
        self.batch = {'half' : np.array(data[0]),
                      'quarter' : np.array(data[1]),
                      'third' : np.array(data[2])
                      } 
    
