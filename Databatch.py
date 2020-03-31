import numpy as np
from matplotlib import pyplot as plt

class DataBatch():
    def __init__(self, data, batch_num, diff, speed, element, category = 'train', damage_state = 1):
        self.data = data
        self.category = category
        self.n_steps = np.shape(self.data)[1]
        self.speed = speed
        self.element = element/180
        self.damage_state = damage_state/100
        assert self.element <= 1, self.damage_state <= 1 
        self.batch = {'1/18' : np.array(data[0]),
                      '1/4'  : np.array(data[1]),
                      '1/2'  : np.array(data[2]),
                      '3/4'  : np.array(data[3]),
                      '17/18': np.array(data[4])
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
        


