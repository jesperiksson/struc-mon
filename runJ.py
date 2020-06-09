# Class files
from Databatch import *
# Utility file
from util import *

if __name__ == "__main__":
    # Which model to use (MLP or LSTM):
    #####################
    use = 'LSTM'
    name = '1'
    #####################

    architecture = {
        'name' :use + name,
        'active_sensors' : ['90'],
        'predict' : 'accelerations', # accelerations or damage
        'path' : 'our_measurements3/e90/'
        }
    sensor_dict = {}
    for i in range(len(architecture['active_sensors'])):
        sensor_dict.update({
            architecture['active_sensors'][i] : i
            })
    architecture.update({
        'sensors' : sensor_dict
        })
    if use == 'MLP':
        from MLP import *
        architecture.update({
            'model' : 'single_layer',
            # Net configuration
            'bias' : True,
            'n_pattern_steps' : 500, # Kan ändras
            'n_target_steps' : 25,
            'pattern_delta' : 10,
            'delta' : 1,
            'n_units' : {'first' : 150, 'second' : 15},
            'loss' : 'rmse',
            # Sensor parameters
            'pattern_sensors' : ['90'], # Indices must be used rahter than placements
            'target_sensor' : '90',
            'target_sensors' : ['90'],
            # Training parameters
            'Dense_activation' : 'tanh',
            'epochs' : 50,
            'patience' : 10,
            'early_stopping' : True,
            'learning_rate' : 0.001, # 0.001 by default
            'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
            'preprocess_type' : 'peaks',      
            'batch_size' : 25,
            # Model saving
            'save_periodically' : True,
            'save_interval' : 10, # Number of series to train on before saving
            # Data interval
            'from' : 0,
            'to' : -1,
            # Classification
            'limit' : 0.9,
            # Plotting
            'metric' : 'rmse'
        })
    elif use == 'LSTM':
        from LSTM import *
        architecture.update({
            'model' : 'single_layer',
            # Net configuaration
            'n_units' : {'first' : 150, 'second' : 100},
            'bias' : True,
            'n_pattern_steps' : 250, # Kan ändras
            'n_target_steps' : 100,
            'pattern_delta' : 25,
            # Sensor parameters
            'pattern_sensors' : ['90'],
            'target_sensor' : '90',
            'target_sensors' : ['90'],
            # Training parameters
            'batch_size' : 10,
            'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
            'delta' : 1, # Kan ändras
            'Dense_activation' : 'tanh',
            'early_stopping' : True,
            'epochs' : 200,
            'learning_rate' : 0.001, # 0.001 by default
            'min_delta' : 0.01,
            'LSTM_activation' : 'tanh',
            'preprocess_type' : 'peaks',
            'patience' : 15,
            # Data interval
            'from' : 0,
            'to' : -1,
            # Model saving
            'save_periodically' : True,
            'save_interval' : 10, # Number of series to train on before saving
            # Classification
            'limit' : 0.9
        })
    elif use == 'AELSTM':
        from AELSTM import *
        architecture.update({
            'n_units' : {'first': 800, 'second': 200, 'third' : 40, 'fourth': 20},
            'bias' : True,
            'speeds' : 20,
            'epochs' : 10,
            'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
            'preprocess_type' : 'data',
            'delta' : 1, # Kan ändras
            'n_pattern_steps' : 400, # Kan ändras
            'batch_size' : 16,
            'n_target_steps' : 400,
            'pattern_delta' : 50,
            'Dense_activation' : 'tanh',
            'LSTM_activation' : 'tanh',
            'learning_rate_schedule' : False,
            'pattern_sensors' : ['90'], 
            'target_sensor' : '90',
            'target_sensors' : ['90'],
            'learning_rate' : 0.01, # 0.001 by default
            'early_stopping' : True,
            'latent_dim' : {'first' : 400, 'second' : 200, 'third' : 40}, 
            'from' : 0,
            'to' : -1,
            # Model saving
            'save_periodically' : True,
            'save_interval' : 10 # Number of series to train on before saving
        })
   
    if architecture['predict'] == 'damage':
        train_series_stack = get_train_series(architecture)
        eval_train_series = train_series_stack
        '''
        This is the normal case where alla available data is divided into train/ test/ validation
        '''

    elif architecture['predict'] == 'accelerations':
        train_series_stack = fit_to_NN(
            architecture['data_split'],
            architecture['path'] + '100%/',#2/e90/100%/'
            100,
            architecture)
        eval_series_stack = get_eval_series(
            {'train':0, 'validation':0, 'test':100}, 
            architecture)
        '''
        This is special case where only healthy data is used for training and 
        all damaged data is used for testing.
        '''
    machine_stack = {}
    
    for i in range(len(architecture['target_sensors'])):
        architecture['target_sensor'] = architecture['target_sensors'][i]
        name = architecture['name']+architecture['target_sensor']
        try:
            f = open('models/'+name+'.json')
            machine_stack.update({
                name : NeuralNet(architecture,
                     name,
                     existing_model = True)
            })
        except IOError:    
            machine_stack.update({
                name : NeuralNet(architecture,
                     name,
                     existing_model = False)
            })
            NeuralNet.train(machine_stack[name], train_series_stack)  
            save_model(machine_stack[name].model, name)
            plot_loss(machine_stack[name], name)  
        
        score_stack = {}
        keys = list(eval_series_stack)
        
        for j in range(len(keys)):
            score_stack.update({
                keys[j] : NeuralNet.evaluation(machine_stack[name], eval_series_stack[keys[j]])
            })
        #print(score_stack)    
        
    plot_performance(score_stack, architecture, 'prediction')
    binary_prediction = get_binary_prediction(score_stack, architecture)
    plot_confusion(binary_prediction, name)
    #plot_roc(binary_prediction)
    ########## PREDICTIONS #############
    prediction_score = {}
    for i in range(len(keys)):
        scores = [None]*len(eval_series_stack[keys[i]])
        speeds = [None]*len(eval_series_stack[keys[i]])
        damage_states = [None]*len(eval_series_stack[keys[i]])
        for j in range(len(eval_series_stack[keys[i]])):
            prediction_manual = {
                'series_to_predict' : j,
                'stack' : eval_series_stack[keys[i]]
            }
            #prediction = NeuralNet.prediction(machine_stack[name], prediction_manual)
            #plot_prediction(prediction, prediction_manual, use)
            forecast, tup = NeuralNet.forecast(machine_stack, prediction_manual)
            scores[j] = tup[0]
            speeds[j] = tup[1]
            damage_states[j] = tup[2]
        prediction_score.update({
            keys[i] : {'scores' : scores, 'speeds' : speeds, 'damage_state' : damage_states}           
            })
        plot_forecast(forecast, prediction_manual, architecture)
    plot_performance(prediction_score, architecture, 'forecast')
        plot_performance(
        prediction_score,
        architecture,
        'forecast')
    binary_forecast_prediction = get_binary_prediction(
        prediction_score,
        architecture)
    plot_confusion(
        binary_forecast_prediction,
        name,
        'forecast')


    
