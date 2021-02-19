from dataclasses import dataclass, field
@dataclass()
class Settings:
    name : str = 'demonstration_SLP_multi_out_aug' # Used to identify saved model
    preset : str = 'SLP_multi_step_output_demonstration' # Must be the name of a module in /presets
    use_preset : bool = True
    template : str = 'Single_layer_perceptron' # Only used if preset = False
    sensor : str = 'acc'
    #features : list = field(default_factory= lambda : ['x','y','z']) # To be automated
    target : list = field(default_factory= lambda : ['x'])


