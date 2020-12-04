import tensorflow as tf
from dataclasses import dataclass, field

@dataclass
class Settings_nn():
    input_width : int = 3
    label_width : int = 1
    shift : int = 3
    first_layer_width : int = 3
    output_layer_width : int = 1
    activation_function : str = 'tanh'
    pattern : list = field(default_factory=lambda:['x'])
    label : list = field(default_factory=lambda:['x'])

@dataclass
class Settings_train():
    epochs : int = 1
    batch_size : int = 20
    verbose : int = 1
    loss : str = 'mse'
    optimizer : str = 'Adam'
    metrics : str = 'mae'
 
@dataclass
class Settings_eval():
    batch_size : int = 20
    verbose : int = 1

@dataclass   
class Settings_test():
    batch_size : int = 20
    verbose : int = 1

@dataclass    
class DataSplit():
    train : int = 0.6
    validation : int = 0.2
    test : int = 0.2
    
def set_up_model(arch):
    inp = tf.keras.layers.Input(shape=(arch.input_width,))
    x = tf.keras.layers.Dense(
        units = arch.first_layer_width,
        activation = arch.activation_function)(inp)
    out = tf.keras.layers.Dense(arch.output_layer_width)(x)
    return tf.keras.Model(inputs = inp, outputs = out)
    
    
