import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    kind : str = 'CNN'
    input_time_steps : int = 300
    target_time_steps : int = 300
    shift : int = 0
    layer_widths : list = field(default_factory=lambda:[]) # Same amount as n_layers
    features : list = field(default_factory=lambda:['acc1_ch_x'])
    targets : list = field(default_factory=lambda:['acc1_ch_x'])
    plot_targets : list = field(default_factory=lambda:['acc1_ch_x'])
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
     
@dataclass
class Settings_eval():
    batch_size : int = 32

@dataclass   
class Settings_test():
    batch_size : int = 32


@dataclass    
class DataSplit():
    train : int = 0.6
    validation : int = 0.2
    test : int = 0.2
    
def set_up_model(arch):
    n_features = len(arch.features)
    n_targets = len(arch.targets)
    conv_width = 15
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
        tf.keras.layers.Conv1D(
            filters = 128,
            activation = 'relu', 
            kernel_size = conv_width),
        #tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dense(arch.target_time_steps*n_features,kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([arch.target_time_steps, n_features])
    ])
    return model    
    
    
