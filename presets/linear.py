import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    input_time_steps : int = 10
    target_time_steps : int = 10 # Must be 1
    shift : int = 1
    first_layer_width : int = 1
    activation_function : str = 'tanh'
    features : list = field(default_factory=lambda:['y'])
    targets : list = field(default_factory=lambda:['y'],)
    plot_target : str = 'y'
    

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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])
    return model
