from dataclasses import dataclass, field
@dataclass()
class Settings:
    name : str = 'test_uru' # Used to identify saved model
    preset : str = 'MLP_multi_step_output_uru' # Must be the name of a module in /presets
    use_preset : bool = True
    sensors : list = field(default_factory=lambda:['acc1','acc2'])
    n_samples : int = 5000
    start_date : str = '2020-10-27'
    end_date : str = '2020-10-28'
    test_end_date : str = '2020-10-29'# For SepTrainTest
    normalization : str = 'min-max' # mean or min-max


