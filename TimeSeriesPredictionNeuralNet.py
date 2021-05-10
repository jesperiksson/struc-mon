
import pandas as pd
import random
from Model import NeuralNet
class TimeSeriesPredictionNeuralNet(NeuralNet):
    def __init__(self,settings):
        super().__init__(settings)
        

        
    def plot_history(self): # Plot the training history for each metric
        key_list = list(self.history.history.keys())
        [plt.plot(self.history.history[key]) for key in key_list]
        plt.legend(key_list)
        plt.title(f'Training history for {self.get_name}, trained for {self.settings_train.epochs} epochs. Elaspsed time: {self.toc}')
        plt.xlabel('epoch')
        plt.ylabel('error') 
        plt.savefig(config.saved_path+self.get_name()+''.join(self.settings.sensors))
        plt.show()   
        
    def plot_example(self): # Plot an input-output example
        rand = random.randint(0,1000000)
        self.time_seriess[rand%len(self.time_seriess)].plot(
            plot_cols = self.settings_model.plot_targets,
            model = self.nn)
        plt.show()  
