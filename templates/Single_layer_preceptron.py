from Base import Base

class Single_layer_perceptron(Base):
    def __init__(self):
        super().__init__()
        
    def set_up_model(self):
        inp = tf.keras.layers.Input(shape=(arch.input_size,))
        x = tf.keras.layers.Dense(
            units = arch.nodes,
            activation = arch.activation_function)(inp)
        out = tf.keras.layers.Dense(arch.output_size)(x)
        return tf.keras.Model(inputs = inp, outputs = out)
    
class settings_nn(Single_layer_perceptron):
    def __init__(self):
        super().__init__()
    
    def set_settings(self):
        self.input_size = self.make_choice('Input size: ')
        self.output_size = self.make_choice('Input size: ')
        self.shift = self.make_choice('Shift: ')
        self.nodes = self.make_choice('Number of nodes: ')
        self.pattern = self.make_choice('Pattern variable: ')
        self.label = self.make_choice('Target variable: ')
        self.activation_function = self.make_choice('Activation function: ',default = 'tanh')
        
class Settings_train(Single_layer_perceptron):
    def __init__(self):
        super().__init__()
        
    def set_settings(self):
        self.epochs = self.make_choice('Training epochs: ', default = '10')
        self.batch_size = self.make_choice('Training Batch size: ', default = 20)
        self.verbose = self.make_choice('Verbose (0 or 1): ', default = '1')
        self.loss = self.make_choice('Loss function', default = 'mse')
        self.optimizer = self.make_choice('Optmizer: ', default = 'Adam')
        self.metrics = self.make_choice('Metrics') # TODO: multiple choices
        
class Settings_eval(Single_layer_perceptron):
    def __init__(self):
        super().__init__()
        
    def set_settings(self):
        self.batch_size = self.make_choice('Evaluation Batch Size: ', default = 20)
        self.verbose = self.make_choice('Verbose (0 or 1): ', default = '1')
        
class Settings_test(Single_layer_perceptron):
    def __init__(self):
        super().__init__()
        
    def set_settings(self):
        self.batch_size = self.make_choice('Test Batch Size: ', default = 20)
        self.verbose = self.make_choice('Verbose (0 or 1): ', default = '1')
        
class DataSplit(Base):
    def __init__(self):
        super().__init__()
        
    def __init__(self):
        self.train = 0
        self.validation = 0
        self.test = 0
        while self.train + self.validation + self.test != 1.0:
            self.train = self.make_choice('Train fraction', default = 0.6)
            self.validation = self.make_choice('Validation fraction', default = 0.2)
            self.test = self.make_choice('Test fraction', default = 0.2)
            if self.train + self.validation + self.test != 1.0:
                print('Error, sum must be 1.0')
        
        

    
        
