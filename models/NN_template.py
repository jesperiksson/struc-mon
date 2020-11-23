# Model template
import tensorflow as tf

def set_up_model(): # 
    arch = {
        'input_dim_1' : 3,
        'input_dim_2' : None,
        'layer_1_size' : 4,
        'output_layer_size' : 1
        }
    inp = tf.keras.layers.Input(shape=(arch['input_dim_1'],arch['input_dim_2']))
    x = tf.keras.layers.Dense(arch['layer_1_size'])(inp)
    out = tf.keras.layers.Dense(arch['output_layer_size'])(x)
    return tf.keras.Model(inputs = inp, outputs = out)

