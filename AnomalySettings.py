from dataclasses import dataclass, field
@dataclass 
class AnomalySettings():
    patience : int = 50
    threshold : int = 3
    anomaly_sensor : list = field(default_factory=lambda:['acc1_ch_x','incl_ch_x'])#,'acc1_ch_z'])#acc1_ch_z
    max_filter : int = 12
