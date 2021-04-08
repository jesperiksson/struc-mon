import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    input_time_steps : int = 32
    target_time_steps : int = 1
    shift : int = 0
    n_layers : int = 2
    layer_widths : list = field(default_factory=lambda:[24,16]) # Same amount as n_layers
    activation_function : str = 'tanh'
    features : list = field(default_factory=lambda:['acc1_ch_x','acc1_ch_y'])
    targets : list = field(default_factory=lambda:['distorted'])
    plot_targets : list = field(default_factory=lambda:['strain1_ch_mv0'])
    verbose : int = 1    
    
@dataclass
class Settings_train():
    epochs : int = 40
    batch_size : int = 32
    loss : str = 'binary_crossentropy'
    optimizer : str = 'Adam'
    metrics : str = field(default_factory=lambda:[
        'binary_crossentropy','accuracy','AUC'])
        #,'FalsePositives','FalseNegatives','TrueNegatives','TruePositives'])
    early_stopping : bool = True
    early_stopping_monitor : str = 'accuracy'
    early_stopping_min_delta : int = 0
    early_stopping_patience : int = 2
    early_stopping_verbose : int = 1
    early_stopping_mode : str = 'auto'
    shuffle : bool = True
     
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
        tf.keras.layers.Dense(arch.layer_widths[0], activation = 'relu'),
        #tf.keras.layers.Dense(arch.layer_widths[1]),        
        tf.keras.layers.Dense(1, activation = 'sigmoid'),#, kernel_initializer=tf.initializers.zeros),

    ])
    return model

    
