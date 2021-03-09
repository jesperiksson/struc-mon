import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    input_time_steps : int = 50
    target_time_steps : int = 5
    shift : int = 3
    first_layer_width : int = 6
    activation_function : str = 'tanh'
    features : list = field(default_factory=lambda:['x','y','z'])
    targets : list = field(default_factory=lambda:['z'])
    plot_target : str = 'z'
    verbose : int = 1    
    
@dataclass
class Settings_train():
    epochs : int = 3
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
        tf.keras.layers.LSTM(arch.first_layer_width, return_sequences=False),
        tf.keras.layers.Dense(arch.target_time_steps*n_features),#, kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Reshape([arch.target_time_steps, n_features])

    ])
    return model
    

    
