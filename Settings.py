from dataclasses import dataclass, field
@dataclass()
class Settings:
    name : str = 'test_classifier_par' # Used to identify saved model
    preset : str = 'MLP_multi_step_output_par' # Must be the name of a module in /presets
    classifier : str = 'naive_classifier'
    sensors : list = field(default_factory=lambda:['acc1'])
    n_samples : int = 5000
    start_date : str = '2020-10-27'
    end_date : str = '2020-11-02'
    test_end_date : str = '2020-11-03'# For SepTrainTest
    normalization : str = 'min-max' # mean or min-max
    
@dataclass    
class DataSplit():
    train : int = 0.6
    validation : int = 0.2
    test : int = 0.2


