import tensorflow as tf
from dataclasses import dataclass, field

tf.get_logger().setLevel('ERROR')

@dataclass
class Settings_nn():
    input_time_steps : int = 50
    target_time_steps : int = 5
    shift : int = 3
    first_layer_width : int = 6
    activation_function : str = 'tanh'
    features : list = field(default_factory=lambda:['acc1_x','acc1_y','acc1_z'])
    targets : list = field(default_factory=lambda:['acc1_z'])
    plot_target : str = 'acc1_z'
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
    
def set_up_model(arch):
    n_features = len(arch.features)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(arch.first_layer_width, return_sequences=False),
        tf.keras.layers.Dense(arch.target_time_steps*n_features),#, kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Reshape([arch.target_time_steps, n_features])

    ])
    return model
    

    
