from dataclasses import dataclass, field
@dataclass()
class Settings:
    name : str = 'test_11' # Used to identify saved model
    preset : str = 'CNN_rsa' # Must be the name of a module in /presets
    classifier : str = 'naive_classifier'
    sensors : list = field(default_factory=lambda:['acc1'])
    n_samples : int = 5000
    start_date : str = '2020-12-27'
    end_date : str = '2020-12-28'
    test_end_date : str = '2020-11-03'# For SepTrainTest
    normalization : str = 'mean' # mean or min-max
    dataset_name : str = '11-01_11-05'
    
@dataclass    
class DataSplit():
    train : int = 0.6
    validation : int = 0.2
    test : int = 0.2


