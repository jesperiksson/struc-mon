from dataclasses import dataclass, field
@dataclass()
class Settings:
    name : str = 'test_1' # Used to identify saved model
    preset : str = 'SLP_single_step' # Must be the name of a module in /models
    use_preset : bool = True
    template : str = 'Single_layer_perceptron' # Only used if preset = False
    sensor : str = 'Acc'
    #features : list = field(default_factory= lambda : ['x','y','z']) # To be automated
    target : list = field(default_factory= lambda : ['x'])


