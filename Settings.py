from dataclasses import dataclass, field
@dataclass()
class Settings:
    name : str = 'test' # Used to identify saved model
    preset : str = 'RNN_test' # Must be the name of a module in /models
    use_preset : bool = True
    template : str = 'Single_layer_perceptron' # Only used if preset = False
    features : list = field(default_factory= lambda : ['x','y','z']) # To be automated
    target : list = field(default_factory= lambda : ['x'])
    


