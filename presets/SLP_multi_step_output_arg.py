import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    input_time_steps : int = 10
    target_time_steps : int = 3
    shift : int = 3
    first_layer_width : int = 6
    activation_function : str = 'tanh'
    features : list = field(default_factory=lambda:['acc1_ch_x','acc1_ch_y','acc1_ch_z','sin_day','cos_day'])
    targets : list = field(default_factory=lambda:['acc1_ch_z'])
    plot_target : str = 'acc1_ch_z'
    verbose : int = 1    
    
@dataclass
class Settings_train():
    epochs : int = 4
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
    train : int = 0.7
    validation : int = 0.15
    test : int = 0.15
    
def set_up_model(arch):
    n_features = len(arch.features)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(arch.target_time_steps*n_features),#, kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Reshape([arch.target_time_steps, n_features])

    ])
    return model
    
