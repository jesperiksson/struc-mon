from dataclasses import dataclass, field
@dataclass()
class Settings:
    name : str = 'demonstration_MLP_aug' # Used to identify saved model
    preset : str = 'MLP_multi_step_demonstration' # Must be the name of a module in /models
    use_preset : bool = True
    template : str = 'Single_layer_perceptron' # Only used if preset = False
    sensor : str = 'acc'
    #features : list = field(default_factory= lambda : ['x','y','z']) # To be automated
    target : list = field(default_factory= lambda : ['x'])


