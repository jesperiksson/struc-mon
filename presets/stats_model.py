from dataclasses import dataclass, field
import tensorflow_probability as tfp

@dataclass
class Settings_model():
    input_time_steps : int = 50
    target_time_steps : int = 0
    shift : int = 0
    features : list = field(default_factory=lambda:['acc1_ch_x','acc1_ch_y','acc1_ch_z'])
    targets : list = field(default_factory=lambda:['acc1_ch_z'])
    plot_target : str = 'acc1_ch_z'
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

    
    return 

