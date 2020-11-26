import tensorflow as tf
settings_nn = {
    'input_width' : 3,
    'label_width' : 1,
    'shift' : 3,
    'first_layer_width' : 3,
    
    'output_layer_width' : 1,
    'activation_function' : 'tanh',
    'input' : ['x'],
    'label' : ['X']
    }
  
settings_train = {
    'epochs' : 1,
    'batch_size' : 20,
    'verbose' : 1,
    'loss' : tf.losses.MeanSquaredError(),
    'optimizer' : tf.optimizers.Adam(),
    'metrics' : [tf.metrics.MeanAbsoluteError()]
    }
    
settings_eval = {
    'batch_size' : 20,
    'verbose' : 1
    }
    
settings_test = {
    'batch_size' : 20,
    'verbose' : 1
    }
    
def set_up_model(arch):
    inp = tf.keras.layers.Input(shape=(settings_nn['input_width'],))
    x = tf.keras.layers.Dense(
        units = settings_nn['first_layer_width'],
        activation = settings_nn['activation_function'])(inp)
    out = tf.keras.layers.Dense(settings_nn['output_layer_width'])(x)
    return tf.keras.Model(inputs = inp, outputs = out)
