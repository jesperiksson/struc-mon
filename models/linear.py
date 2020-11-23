import tensorflow as tf
settings_nn = {
    'input_width' : 3,
    'label_width' : 1,
    'shift' : 3,
    'first_layer_width' : 3,
    
    'output_layer_width' : 1,
    'activation_function' : None,
    'input' : ['x'],
    'label' : ['X']
    }
  
settings_train = {
    'epochs' : 5,
    'batch_size' : 10,
    'verbose' : 1,
    'loss' : tf.losses.MeanSquaredError(),
    'optimizer' : tf.optimizers.Adam(),
    'metrics' : [tf.metrics.MeanAbsoluteError()]
    }
    
def set_up_model(arch):
    return tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
