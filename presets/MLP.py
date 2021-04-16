import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    kind : str = 'MLP'
    input_time_steps : int = 10
    target_time_steps : int = 2
    shift : int = 2
    n_layers : int = 2
    layer_widths : list = field(default_factory=lambda:[12]) # Same amount as n_layers
    features : list = field(default_factory=lambda:['max'])
    targets : list = field(default_factory=lambda:['max'])
    plot_targets : list = field(default_factory=lambda:['max'])
    verbose : int = 1    
    
@dataclass
class Settings_train():
    epochs : int = 40
    batch_size : int = 32
    loss : str = 'mse'
    optimizer : str = 'Adam'
    metrics : str = field(default_factory=lambda:['mae'])
    early_stopping : bool = True
    early_stopping_monitor : str = 'loss'
    early_stopping_min_delta : int = 0
    early_stopping_patience : int = 0
    early_stopping_verbose : int = 1
    early_stopping_mode : str = 'auto'
    shuffle = True
     
@dataclass
class Settings_eval():
    batch_size : int = 32

@dataclass   
class Settings_test():
    batch_size : int = 32

    

def set_up_model(arch):
    n_features = len(arch.features)
    n_targets = len(arch.targets)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(arch.layer_widths[0],activation = 'relu'),
        #tf.keras.layers.Dense(arch.layer_widths[1]),        
        tf.keras.layers.Dense(arch.target_time_steps*n_targets,activation = 'relu'),
        tf.keras.layers.Reshape([arch.target_time_steps, n_targets])

    ])
    return model

    
