from dataclasses import dataclass, field
@dataclass()
class Settings:
    name : str = 'test_bol_4' # Used to identify saved model
    preset : str = 'MLP_multi_step_output_bol' # Must be the name of a module in /presets
    classifier : str = 'naive_classifier'
    sensors : list = field(default_factory=lambda:['acc1'])
    n_samples : int = 5000
    start_date : str = '2020-11-03'
    end_date : str = '2020-11-04'
    test_end_date : str = '2020-11-03'# For SepTrainTest
    normalization : str = 'mean' # mean or min-max
    
@dataclass    
class DataSplit():
    train : int = 0.6
    validation : int = 0.2
    test : int = 0.2


