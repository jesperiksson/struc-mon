def set_settings(placeholder = False):
    if placeholder == False:
        settings = {
            'name' : 'foobar',
            'model_path' : '',
            # Which neural net
            'model' : 'Single layer MLP',
            # Net configuaration
            'n_units' : {
                'first' : 3, 
                'second' : 2, 
                'third' : 1, 
                'fourth' : 1
                },
            'bias' : True,
            'n_pattern_steps' : 2, # Kan ändras
            'n_target_steps' : 1,
            'pattern_delta' : 1,
            # Sensor parameters
            'pattern_sensors' : ['x','y','z'], # All of them are used in each machine
            'target_sensors' : ['x','y','z'], # One machine for each of them 
            # Training parameters
            'batch_size' : 16,
            'positions' : False,
            'data_split' : {
                'train':30, 
                'validation':20, 
                'test':50
                }, # sorting of data 
            'delta' : 1, # Kan ändras
            'Dense_activation' : 'tanh',
            'early_stopping' : True,
            'epochs' : 200,
            'learning_rate' : 0.001, # 0.001 by default
            'min_delta' : 0.0001,
            'preprocess_type' : 'data',
            'patience' : 10,
            'shuffle' : True,
            'steps_per_epoch' : 100,
            'validation_steps' : 50,
            'dropout_rate' : 0.2, # Only in use if model utilizes dropout
            'features' : 1,
            # Loss function
            'loss' : 'mse',
            'metric': 'rmse',
            'val_loss' : 'val_loss',
            # Data
            'normalization' : 'L-2',
            'speed_unit' : 'km/h',
            'seed' : 3,
            'mode' : '2', # 1 - , 2- 
            'from' : 0,
            'to' : -1,
            'random_mode' : 'debug', # test or debug
            'data_function' : 'one_by_one', # one_by_one or all_at_once
            # Noise
            'noise' : False,
            'noise_mean' : 0,
            'noise_var' : 0.001,
            # Evaluation parameters
            'eval_batch_size' : 32,
            # Model saving
            'save_periodically' : False,
            'save_interval' : 10, # Number of series to train on before saving
            'foi' : 'normalization', # feature of interest, the one to be printed in the title of plots
            # Classification
            'probability_limit' : 0.90,
            'fitting_points' : 20,
            'z-score' : 2.2,
            'confusion_matrices' : {
                'cdf' : 'CDF',
                'poly' : 'Polynomial approximation'
                },
            'poly_deg' : 1
        }
    elif placeholder == True:
        settings = {'name' : 'placeholder'}
    return settings