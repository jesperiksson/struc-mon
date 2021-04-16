from dataclasses import dataclass, field
@dataclass()
class Settings:
    #name : str = 'LSTM_32_6_4_24_x1-x2' # Used to identify saved model
    preset : str = 'MLP' # Must be the name of a module in /presets
    classifier : str = 'naive_classifier'
    sensors : list = field(default_factory=lambda:['acc1','incl'])#,'acc2','incl','strain1'])
    agg_sensor : str = 'acc1_z'
    n_samples : int = 5000
    start_date : str = '2020-11-02'
    end_date : str = '2020-11-03'
    test_end_date : str = '2020-11-03'# For SepTrainTest
    normalization : str = 'mean' # mean or min-max
    dataset_name : str = '03-01_03-05_32_6_4_test'
    
@dataclass    
class DataSplit():
    train : int = 0.7
    validation : int = 0.15
    test : int = 0.15


