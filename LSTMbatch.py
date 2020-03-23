import numpy as np
from matplotlib import pyplot as plt

class LongShortTermMemoryBatch():
    def __init__(self, data, batch_num, diff, category = 'train', damage_state = 'H'):
        self.data = data
        self.category = category
        self.diff = diff
        self.n_steps = np.shape(self.data)[1]
        self.n_sensors = np.shape(self.data)[0]
        self.batch = {'half' : np.array(data[0]),
                      'quarter' : np.array(data[1]),
                      'third' : np.array(data[2])
                      } 

    def plot_batch(self, sensor_list, which_batch = 1):
        fig, axs = plt.subplots(self.n_sensors, 1, constrained_layout = True)
        for i in range(self.n_sensors):
            axs[i].plot(range(self.n_steps), self.data[i], 'b', linewidth=0.4)
            axs[i].set_title(sensor_list[i])
            axs[i].set_xlabel('timestep')
            axs[i].set_ylabel('acceleration')
        plt.suptitle('Batch '+str(which_batch))
        plt.show()
        return
        


