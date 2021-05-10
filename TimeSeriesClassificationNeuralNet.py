
import pandas as pd
from Model import NeuralNet
class TimeSeriesClassificationNeuralNet(NeuralNet): 
    def __init__(self,settings):
        super().__init__(settings)
        
    def make_timeseries_category_dataset(self, data):
        time_seriess = []
        cols = sorted(list(set(self.settings_model.features+self.settings_model.targets)))
        for i in range(len(data.dfs)):      
            time_seriess.append(
                WindowClassificationGenerator(
                    input_width = self.settings_model.input_time_steps,
                    shift = self.settings_model.shift,
                    train_df = data.train_dfs[i][cols],
                    val_df = data.val_dfs[i][cols],
                    test_df = data.test_dfs[i][cols],
                    train_batch_size = self.settings_train.batch_size,
                    eval_batch_size = self.settings_eval.batch_size,
                    test_batch_size = self.settings_test.batch_size)
                )
            
        self.time_seriess = time_seriess
        self.dates = data.dates
        
    def plot_history(self): # Plot the training history for each metric
        key_list = ['binary_crossentropy','accuracy','auc']
        [plt.plot(self.history.history[key]) for key in key_list]
        plt.legend(key_list)
        plt.title(f'Training history for {self.get_name()}, trained for {self.settings_train.epochs} epochs. Elaspsed time: {self.toc}')
        plt.xlabel('epoch')
        plt.ylabel('error') 
        plt.savefig(config.saved_path+self.settings.name+''.join(self.settings.sensors))
        plt.show()   
        
    def plot_auc(self):
        plt.plot(
#            self.history.history['auc'],
#            self.history.history['false_positives']*(1/max(self.history.history['false_positives'])),
#            self.history.history['true_positives']*(1/max(self.history.history['true_positives'])),
            [x/max(self.history.history['false_positives']) for x in self.history.history['false_positives']],
            [x/max(self.history.history['true_positives']) for x in self.history.history['true_positives']],
            color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
