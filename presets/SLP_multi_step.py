import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    input_time_steps : int = 4
    target_time_steps : int = 2
    shift : int = 1
    first_layer_width : int = 3
    activation_function : str = 'tanh'
    features : list = field(default_factory=lambda:['ch1'])
    targets : list = field(default_factory=lambda:['ch1'])
    plot_target : str = 'x'
    verbose : int = 1    
    
@dataclass
class Settings_train():
    epochs : int = 1
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
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(arch.first_layer_width),
        tf.keras.layers.Dense(n_features),
        tf.keras.layers.Reshape([1, -1])
    ])
    return model
