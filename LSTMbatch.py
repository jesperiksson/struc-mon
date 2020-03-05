import numpy as np

class LongShortTermMemoryBatch():
    def __init__(self, data, batch_num, category = 'train', damage_state = 'H'):
        self.data = data
#        print(np.shape(self.data))
        self.category = category
        self.n_steps = np.shape(data[1])
        self.n_files = np.shape(data[0])
        self.batch = np.array(data) 
        
