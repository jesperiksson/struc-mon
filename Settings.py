from dataclasses import dataclass, field
@dataclass()
class Settings:
    name : str = 'demonstration_LSTM_multi_out_aug' # Used to identify saved model
    preset : str = 'LSTM_multi_step_output_strain_demonstration' # Must be the name of a module in /presets
    use_preset : bool = True
    template : str = 'Single_layer_perceptron' # Only used if preset = False
    sensors : list = field(default_factory=lambda:['acc1','acc2','incl'])


