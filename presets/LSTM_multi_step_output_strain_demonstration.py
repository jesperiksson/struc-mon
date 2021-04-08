import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    input_time_steps : int = 50
    target_time_steps : int = 5
    shift : int = 3
    first_layer_width : int = 6
    activation_function : str = 'tanh'
    features : list = field(default_factory=lambda:['acc1_ch_x','acc1_ch_y','acc1_ch_z','sin_day','cos_day'])
    targets : list = field(default_factory=lambda:['acc1_ch_x'])
    plot_targets : list = field(default_factory=lambda:['acc1_ch_x'])
    verbose : int = 1    
    
@dataclass
class Settings_train():
    epochs : int = 40
    batch_size : int = 32
    loss : str = 'mse'
    optimizer : str = 'Adam'
    metrics : str = field(default_factory=lambda:['mse','mae'])
    early_stopping : bool = True
    early_stopping_monitor : str = 'loss'
    early_stopping_min_delta : int = 0
    early_stopping_patience : int = 0
    early_stopping_verbose : int = 1
    early_stopping_mode : str = 'auto'
     
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
    n_targets = len(arch.targets)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(arch.first_layer_width, return_sequences=False),
        tf.keras.layers.Dense(arch.target_time_steps*n_targets),#, kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Reshape([arch.target_time_steps, n_targets])

    ])
    return model
    

    