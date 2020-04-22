import numpy as np
from matplotlib import pyplot as plt

class DataBatch():
    def __init__(self, data, batch_num, speed, normalized_speed, category = 'train'):
        self.data = data
        self.category = category
        self.n_steps = np.shape(self.data)[1]
        self.speed = speed
        self.normalized_speed = normalized_speed 
        self.batch = {'half' : np.array(data[0]),
                      'quarter'  : np.array(data[1]),
                      'third'  : np.array(data[2])
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
        


