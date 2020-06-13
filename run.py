# Class files
from Databatch import *
# Utility file
from util import *

if __name__ == "__main__":
    
    #####################
    use = 'LSTM'                    # Which model to use (MLP or LSTM):
    name = 'test'                   # Name of model
    overwrite = False               # True: erase any model with that name and make a new model
                                    # False: Try to load a model with that name, if none exists
                                    # a new one is saved. 
    #####################  
    if overwrite == False:
        try: 
            architecture = load_architecture(name)
            print('Loaded architecture: ', name)
            new_model = False
            if use == 'MLP':
                from MLP import *
            elif use == 'LSTM':
                from LSTM import *
            elif use == 'AELSTM':
                from AELSTM import *
        except IOError: 
            new_model = True
    elif overwrite == True:    
        new_model = True
         
    if new_model == True:
        architecture = {
            'name' :use + name,
            'active_sensors' : ['90'],
            'predict' : 'accelerations', # accelerations or damage
            'path' : 'our_measurements3/e90/',
            'random_mode' : 'test' # test or debug
            }
        sensor_dict = {}
        for i in range(len(architecture['active_sensors'])):
            sensor_dict.update({
                architecture['active_sensors'][i] : i
                })
        architecture.update({
            'sensors' : sensor_dict
            })
        plotting = {
            'prediction_performance' : True,
            'prediction_confusion_matrix' : True,
            'prediction_roc' : False, # To be implemented
            'forecast_performance' : True,
            'forecast_confusion_matrix' : True
            }

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
                'pattern_sensors' : ['45'], 
                'target_sensor' : '45',
                'target_sensors' : ['45'],
                # Training parameters
                'Dense_activation' : 'tanh',
                'epochs' : 50,
                'patience' : 10,
                'early_stopping' : True,
                'learning_rate' : 0.001, # 0.001 by default
                'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
                'mode' : '1',
                'preprocess_type' : 'data',      
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
                'model' : 'two_layer',
                # Net configuaration
                'n_units' : {'first' : 300, 'second' : 150, 'third' : 100},
                'bias' : True,
                'n_pattern_steps' : 200, # Kan ändras
                'n_target_steps' : 10,
                'pattern_delta' : 4,
                # Sensor parameters
                'pattern_sensors' : ['90'],
                'target_sensor' : '90',
                'target_sensors' : ['90'],
                # Training parameters
                'batch_size' : 10,
                'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
                'mode' : '1',
                'delta' : 1, # Kan ändras
                'Dense_activation' : 'tanh',
                'recurrent_activation' : 'sigmoid',
                'early_stopping' : True,
                'epochs' : 200,
                'learning_rate' : 0.001, # 0.001 by default
                'min_delta' : 0.01,
                'LSTM_activation' : 'tanh',
                'preprocess_type' : 'peaks',
                'patience' : 12,
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
                'model' : 'AE',
                # Net configuaration
                'n_units' : {'first' : 200, 'second' : 150, 'third' : 100},
                'latent_dim' : {'first' : 200, 'second' : 150, 'third' : 125},
                'bias' : True,
                'in_out_put_size' : 200, # Kan ändras
                'pattern_delta' : 1,
                # Sensor parameters
                'pattern_sensors' : ['90'],
                'target_sensor' : '90',
                'target_sensors' : ['90'],
                # Training parameters
                'batch_size' : 10,
                'data_split' : {'train':60, 'validation':20, 'test':20}, # sorting of data 
                'mode' : '1',
                'delta' : 1, # Kan ändras
                'Dense_activation' : 'tanh',
                'early_stopping' : True,
                'epochs' : 50,
                'learning_rate' : 0.001, # 0.001 by default
                'min_delta' : 0.01,
                'LSTM_activation' : 'tanh',
                'preprocess_type' : 'peaks',
                'patience' : 15,
                'scheduled_sampling' : False,
                'sampling_rate' : 0.2,
                # Data interval
                'from' : 0,
                'to' : -1,
                # Model saving
                'save_periodically' : True,
                'save_interval' : 10, # Number of series to train on before saving
                # Classification
                'limit' : 0.9
            })
        save_architecture(architecture, name)
   
    if architecture['mode'] == '1':
        series_stack = data_split_mode1(architecture)
        '''
        This is the normal case where all available data is divided into train/ test/ validation
        '''

    elif architecture['mode'] == '2':
        series_stack = data_split_mode2(architecture)
        '''
        This is special case where only healthy data is used for training and 
        all damaged data is used for testing.
        '''
    machine_stack = {}
    
    for i in range(len(architecture['target_sensors'])):
        architecture['target_sensor'] = architecture['target_sensors'][i]
        name = architecture['name']
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
            NeuralNet.train(machine_stack[name], series_stack)  
            save_model(machine_stack[name].model, name)
            plot_loss(machine_stack[name], name)  
        
        score_stack = {}
        keys = list(series_stack)
        
        for j in range(len(keys)):
            score_stack.update({
                keys[j] : NeuralNet.evaluation(machine_stack[name], series_stack[keys[j]])
            })
    if plotting['prediction_performance'] == True:  
        plot_performance(score_stack, architecture, 'prediction')
    if plotting['prediction_confusion_matrix'] == True:
        binary_prediction = get_binary_prediction(score_stack, architecture)
        plot_confusion(binary_prediction, name,'prediction')
    if plotting['prediction_roc'] == True:
        plot_roc(binary_prediction)
    ########## PREDICTIONS #############
    prediction_score = {}
    for i in range(len(keys)):
        #print(series_stack[keys[i]])
        scores = []
        speeds = []
        damage_states = []
        for j in range(len(series_stack[keys[i]][architecture['preprocess_type']])):
            if series_stack[keys[i]][architecture['preprocess_type']]['batch'+str(j)].category == 'test':
                prediction_manual = {
                    'series_to_predict' : j,
                    'stack' : series_stack[keys[i]]
                }
                #prediction = NeuralNet.prediction(machine_stack[name], prediction_manual)
                #plot_prediction(prediction, prediction_manual, use)
                forecast, tup = NeuralNet.forecast(machine_stack, prediction_manual)
                scores.extend([tup[0]])
                speeds.extend([tup[1]])
                damage_states.extend([tup[2]])
            else:
                continue
        prediction_score.update({
            keys[i] : {'scores' : scores, 'speeds' : speeds, 'damage_state' : damage_states}           
            })
        plot_forecast(forecast, prediction_manual, architecture)
    if plotting['forecast_performance'] == True:
        plot_performance(
            prediction_score,
            architecture,
            'forecast')
    if plotting['forecast_confusion_matrix'] == True:
        binary_forecast_prediction = get_binary_prediction(
            prediction_score,
            architecture)
        plot_confusion(
            binary_forecast_prediction,
            name,
            'forecast')


    
