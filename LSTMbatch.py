import numpy as np
from matplotlib import pyplot as plt

class LongShortTermMemoryBatch():
    def __init__(self, data, batch_num, category = 'train', damage_state = 'H'):
        self.data = data
        #print(np.shape(self.data))
        self.category = category
        self.n_steps = np.shape(self.data)[1]
        self.n_sensors = np.shape(self.data)[0]
        self.batch = np.array(data) 

    def plot_batch(self, sensor_dict, which_batch = 1):
        fig, axs = plt.figure()
        for i in range(self.n_sensors):
            ax = fig.add_subplot(self.n_sensors, 1, i+1)
            ax.plot(range(self.n_steps), self.data[i], 'b', linewidth=0.1, label=sensor_dict.get(i))
            ax.set_title(sensor_dict.get(i))
        plt.suptitle('Batch '+str(which_batch))
        plt.legend()
        plt.show()
        return
        
