import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    input_time_steps : int = 50
    target_time_steps : int = 10
    shift : int = 10
    n_layers : int = 2
    layer_widths : list = field(default_factory=lambda:[128,64]) # Same amount as n_layers
    activation_function : str = 'tanh'
    features : list = field(default_factory=lambda:['acc1_ch_x','acc1_ch_y','acc1_ch_z','sin_day','cos_day'])
    targets : list = field(default_factory=lambda:['acc1_ch_x','acc1_ch_y','acc1_ch_z'])
    plot_target : str = 'acc1_ch_z'
    verbose : int = 1    
    
@dataclass
class Settings_train():
    epochs : int = 11
    batch_size : int = 64
    loss : str = 'mse'
    optimizer : str = 'Adam'
    metrics : str = 'mae'
     
@dataclass
class Settings_eval():
    batch_size : int = 64

@dataclass   
class Settings_test():
    batch_size : int = 64


@dataclass    
class DataSplit():
    train : int = 0.6
    validation : int = 0.2
    test : int = 0.2
    
    
def set_up_model(arch):
    n_features = len(arch.features)
    n_targets = len(arch.targets)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(arch.layer_widths[0]),
        tf.keras.layers.Dense(arch.target_time_steps*n_targets),#, kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Reshape([arch.target_time_steps, n_targets])

    ])
    return model
    
    
    
