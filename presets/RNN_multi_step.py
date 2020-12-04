import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    input_time_steps : int = 4
    target_time_steps : int = 2
    shift : int = 1
    first_layer_width : int = 3
    #output_layer_width : int = 2
    activation_function : str = 'tanh'
    features : list = field(default_factory=lambda:['x'])
    targets : list = field(default_factory=lambda:['x'],)
    plot_target : str = 'x'
    
    @property
    def n_features(self):
        return len(self.features)
    
    @property
    def n_targets(self):
        return len(self.targets)
    

@dataclass
class Settings_train():
    epochs : int = 1
    batch_size : int = 20
    verbose : int = 1
    loss : str = 'mse'
    optimizer : str = 'Adam'
    metrics : str = 'mae'
 
@dataclass
class Settings_eval():
    batch_size : int = 20
    verbose : int = 1

@dataclass   
class Settings_test():
    batch_size : int = 20
    verbose : int = 1

@dataclass    
class DataSplit():
    train : int = 0.6
    validation : int = 0.2
    test : int = 0.2
    
def set_up_model(arch):
    n_features = len(arch.features)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(arch.first_layer_width, return_sequences = False),
        tf.keras.layers.Dense(n_features*arch.target_time_steps),
        tf.keras.layers.Reshape([arch.target_time_steps, n_features])
    ])
    return model
