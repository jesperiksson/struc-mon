from dataclasses import dataclass, field
@dataclass()
class Settings:
    name : str = 'test_NN'
    model : str = 'test_NN'
    preset : bool = True
    features : list = field(default_factory= lambda : ['x','y','z']) # To be automated
    target : list = field(default_factory= lambda : ['x'])
    


