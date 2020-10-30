"""# Models

##Single layer LSTM CPU
"""
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, LSTM, CuDNNLSTM, concatenate, Activation, Reshape, Flatten, Dropout
def set_up_model6(arch): # Vanilla 

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            1),
        name = 'accel_input_90')

    hidden_lstm_1 = CuDNNLSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        #activation = 'tanh',
        #recurrent_activation = 'sigmoid',
        #use_bias = True,
        #dropout = 0.1,
        #recurrent_dropout = 0,
        #unroll = False,
        return_sequences = True,
        stateful = False)(accel_input)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_1)

    model = Model(inputs = accel_input, outputs = output)

    return model

"""## Single layer LSTM non-CPU"""

def set_up_model1(arch): # Vanilla 

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            1),
        name = 'accel_input_90')

    hidden_lstm_1 = LSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        #activation = 'tanh',
        #recurrent_activation = 'sigmoid',
        #use_bias = True,
        #dropout = 0.1,
        #recurrent_dropout = 0,
        #unroll = False,
        return_sequences = True,
        stateful = False)(accel_input)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_1)

    model = Model(inputs = accel_input, outputs = output)

    return model

"""## Two-layer LSTM CPU with positions"""

def set_up_model3(arch):

    accel_input = Input(
        shape = (
            arch['n_pattern_steps'], 
            arch['features']),
        name = 'accel_input_90')


    hidden_lstm_1 = CuDNNLSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        #activation = 'tanh',
        #recurrent_activation = 'sigmoid',
        #use_bias = True,
        #dropout = 0.1,
        #recurrent_dropout = 0,
        #unroll = False,
        return_sequences = True,
        stateful = False)(accel_input)

    hidden_lstm_2 = CuDNNLSTM(
        arch['n_units']['second'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        #activation = 'tanh',
        #recurrent_activation = 'sigmoid',
        #use_bias = True,
        #dropout = 0.1,
        #recurrent_dropout = 0,
        #unroll = False,
        stateful = False)(hidden_lstm_1)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_2)

    model = Model(inputs = accel_input, outputs = output)

    return model

"""## Two-layer LSTM CPU"""

def set_up_model7(arch):

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            1),
        name = 'accel_input_90')

    hidden_lstm_1 = CuDNNLSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        return_sequences = True,
        stateful = False)(accel_input)

    hidden_lstm_2 = CuDNNLSTM(
        arch['n_units']['second'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        stateful = False)(hidden_lstm_1)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_2)

    model = Model(inputs = accel_input, outputs = output)

    return model

"""## Single-layer LSTM CPU arbitrary sensors"""

def set_up_model8(arch):

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            arch['features']),
        name = 'accel_input_90')

    hidden_lstm_1 = CuDNNLSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        return_sequences = False,
        stateful = False)(accel_input)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_1)

    model = Model(inputs = accel_input, outputs = output)

    return model

"""## Two-layer LSTM CPU arbitrary sensors"""

def set_up_model4(arch):

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            arch['features']),
        name = 'accel_input_90')

    hidden_lstm_1 = CuDNNLSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        return_sequences = True,
        stateful = False)(accel_input)

    hidden_lstm_2 = CuDNNLSTM(
        arch['n_units']['second'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        stateful = False)(hidden_lstm_1)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_2)

    model = Model(inputs = accel_input, outputs = output)

    return model

"""## Three-layer LSTM CPU arbitrary sensors"""

def set_up_model9(arch):

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            arch['features']),
        name = 'accel_input_90')

    hidden_lstm_1 = CuDNNLSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        return_sequences = True,
        stateful = False)(accel_input)

    hidden_lstm_2 = CuDNNLSTM(
        arch['n_units']['second'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        return_sequences = True,
        stateful = False)(hidden_lstm_1)

    hidden_lstm_3 = CuDNNLSTM(
        arch['n_units']['third'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        stateful = False)(hidden_lstm_2)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_3)

    model = Model(inputs = accel_input, outputs = output)

    return model

"""## Four-layer LSTM CPU arbitrary sensors"""

def set_up_model12(arch):

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            arch['features']),
        name = 'accel_input_90')

    hidden_lstm_1 = CuDNNLSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        return_sequences = True,
        stateful = False)(accel_input)

    hidden_lstm_2 = CuDNNLSTM(
        arch['n_units']['second'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        return_sequences = True,
        stateful = False)(hidden_lstm_1)

    hidden_lstm_3 = CuDNNLSTM(
        arch['n_units']['third'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        return_sequences = True,
        stateful = False)(hidden_lstm_2)

    hidden_lstm_4 = CuDNNLSTM(
        arch['n_units']['fourth'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        return_sequences = False,
        stateful = False)(hidden_lstm_3)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_4)

    model = Model(inputs = accel_input, outputs = output)

    return model

"""## Two-layer LSTM non-CPU"""

def set_up_model2(arch):

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            1),
        name = 'accel_input_90')

    hidden_lstm_1 = LSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        activation = 'tanh',
        recurrent_activation = 'sigmoid',
        use_bias = True,
        dropout = 0.1,
        recurrent_dropout = 0,
        unroll = False,
        return_sequences = True,
        stateful = False)(accel_input)

    hidden_lstm_2 = LSTM(
        arch['n_units']['second'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            1),
        activation = 'tanh',
        recurrent_activation = 'sigmoid',
        use_bias = True,
        dropout = 0.1,
        recurrent_dropout = 0,
        unroll = False,
        stateful = False)(hidden_lstm_1)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_2)

    model = Model(inputs = accel_input, outputs = output)

    return model

"""## Single-layer MLP"""

def set_up_model10(arch):
    
    accel_input = Input(
        shape = (
            arch['n_pattern_steps'],
            arch['features']),
        name = 'accel_input_'+arch['target_sensor'])
    
    flat = Flatten()(accel_input)

    hidden_1 = Dense(
        arch['n_units']['first'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'],
        name = 'hidden_layer')(flat)

    output = Dense(
        arch['n_target_steps'], 
        activation = arch['Dense_activation'], 
        name='acceleration_output')(hidden_1) 

    model = Model(inputs = accel_input, outputs = output)
    return model

"""## Two-layer MLP"""

def set_up_model5(arch):
    
    accel_input = Input(
        shape = (
            arch['n_pattern_steps'],
            arch['features']),
        name = 'accel_input_'+arch['target_sensor'])
    
    flat = Flatten()(accel_input)

    hidden_1 = Dense(
        arch['n_units']['first'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'],
        name = 'hidden_layer')(flat)

    hidden_2 = Dense(
        arch['n_units']['second'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'])(hidden_1)

    output = Dense(
        arch['n_target_steps'], 
        activation = 'tanh', 
        name = 'acceleration_output')(hidden_2)

    model = Model(inputs = accel_input, outputs = output)
    return model

"""## Three-layer MLP"""

def set_up_model13(arch):
    
    accel_input = Input(
        shape = (
            arch['n_pattern_steps'],
            arch['features']),
        name = 'accel_input_'+arch['target_sensor'])
    
    flat = Flatten()(accel_input)

    hidden_1 = Dense(
        arch['n_units']['first'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'],
        name = 'hidden_layer')(flat)

    hidden_2 = Dense(
        arch['n_units']['second'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'])(hidden_1)

    hidden_3 = Dense(
        arch['n_units']['third'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'])(hidden_2)

    output = Dense(
        arch['n_target_steps'], 
        activation = 'tanh', 
        name = 'acceleration_output')(hidden_3)

    model = Model(inputs = accel_input, outputs = output)
    return model

"""## Four-layer MLP"""

def set_up_model11(arch):
    
    accel_input = Input(
        shape = (
            arch['n_pattern_steps'],
            arch['features']),
        name = 'accel_input_'+arch['target_sensor'])
    
    flat = Flatten()(accel_input)

    hidden_1 = Dense(
        arch['n_units']['first'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'],
        name = 'hidden_layer')(flat)

    hidden_2 = Dense(
        arch['n_units']['second'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'])(hidden_1)

    hidden_3 = Dense(
        arch['n_units']['third'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'])(hidden_2)

    hidden_4 = Dense(
        arch['n_units']['fourth'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'])(hidden_3)

    output = Dense(
        arch['n_target_steps'], 
        activation = 'tanh', 
        name = 'acceleration_output')(hidden_4)

    model = Model(inputs = accel_input, outputs = output)
    return model

"""## Single-layer MLP dropout"""

def set_up_model14(arch):
    
    accel_input = Input(
        shape = (
            arch['n_pattern_steps'],
            arch['features']),
        name = 'accel_input_'+arch['target_sensor'])
    
    dropout = Dropout(
        arch['dropout_rate'],
        name = 'dropout')(accel_input)
    
    flat = Flatten()(dropout)

    hidden_1 = Dense(
        arch['n_units']['first'],
        activation = arch['Dense_activation'],
        use_bias = arch['bias'],
        name = 'hidden_layer')(flat)

    output = Dense(
        arch['n_target_steps'], 
        activation = arch['Dense_activation'], 
        name='acceleration_output')(hidden_1) 

    model = Model(inputs = accel_input, outputs = output)
    return model

"""## Single-layer LSTM CPU arbitrary sensors"""

def set_up_model15(arch):

    accel_input = Input(
        shape=(
            arch['n_pattern_steps'], 
            arch['features']),
        name = 'accel_input_90')
    
    dropout = Dropout(
        arch['dropout_rate'],
        name = 'dropout')(accel_input)

    hidden_lstm_1 = CuDNNLSTM(
        arch['n_units']['first'],
        batch_input_shape = (
            arch['batch_size'],
            arch['n_pattern_steps'],
            arch['features']),
        return_sequences = False,
        stateful = False)(dropout)

    output = Dense(
        arch['n_target_steps'], 
        activation='tanh', 
        name='peak_output_90')(hidden_lstm_1)

    model = Model(inputs = accel_input, outputs = output)

    return model
