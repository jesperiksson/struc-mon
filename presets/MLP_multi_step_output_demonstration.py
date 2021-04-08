import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    input_time_steps : int = 10
    target_time_steps : int = 3
    shift : int = 1
    n_layers : int = 3
    layer_widths : list = field(default_factory=lambda:[8,7,6]) # Same amount as n_layers
    activation_function : str = 'tanh'
    features : list = field(default_factory=lambda:['z'])
    targets : list = field(default_factory=lambda:['z'])
    plot_target : str = 'z'
    verbose : int = 1    
    
@dataclass
class Settings_train():
    epochs : int = 5
    batch_size : int = 20
    loss : str = 'mse'
    optimizer : str = 'Adam'
    metrics : str = 'mae'
     
@dataclass
class Settings_eval():
    batch_size : int = 20

@dataclass   
class Settings_test():
    batch_size : int = 20


@dataclass    
class DataSplit():
    train : int = 0.6
    validation : int = 0.2
    test : int = 0.2
    
def set_up_model(arch):
    n_features = len(arch.features)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(arch.layer_widths[0]))
    for i in range(arch.n_layers-1):
        model.add(tf.keras.layers.Dense(arch.layer_widths[i+1]))
    model.add(tf.keras.layers.Dense(arch.target_time_steps))
    model.add(tf.keras.layers.Reshape([1, -1]))
    return model
    
    
