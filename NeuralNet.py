"""# `NeuralNet` class
Most importantly contains the `self.model` attribute. The `model_dict` dictionary variable must be updated once a new model settingsitecture is added.
"""

class NeuralNet():
    def __init__(
        self,
        settings,
        existing_model):

        self.settings = settings
        self.name = settings['name']
        self.target_sensor = self.settings['sensors'][self.settings['target_sensor']]
        self.pattern_sensors = self.settings['sensors'][self.settings['pattern_sensors'][0]]
        self.sensor_to_predict = settings['sensors'][settings['target_sensor']]
        if settings['early_stopping'] == True:
            self.early_stopping = [keras.callbacks.EarlyStopping(
                monitor = settings['val_loss'],
                min_delta = settings['min_delta'], 
                patience = settings['patience'],
                verbose = 1,
                mode = 'auto',
                restore_best_weights = True)]

        else:
            self.early_stopping = None
        self.existing_model = existing_model
        self.n_sensors = len(settings['sensors'])    
        model_dict = {
            'Single layer LSTM-CPU' : set_up_model6(settings),
            'Two layer CPU-LSTM' : set_up_model7(settings),
            'Single layer LSTM-non-CPU' : set_up_model1(settings), 
            'Two layer LSTM-non-CPU' : set_up_model2(settings),
            'Two layer LSTM CPU position' : set_up_model3(settings),
            'Single layer LSTM' : set_up_model8(settings),
            'Two layer LSTM' : set_up_model4(settings),
            'Three layer LSTM' : set_up_model9(settings),
            'Four layer LSTM' : set_up_model12(settings),
            'Single layer MLP' : set_up_model10(settings),
            'Two layer MLP' : set_up_model5(settings),
            'Three layer MLP' : set_up_model13(settings),
            'Four layer MLP' : set_up_model11(settings),
            'Single layer MLP dropout' : set_up_model14(settings),
            'Single layer LSTM dropout' : set_up_model15(settings)
            }     
        metric_dict = {
            'rmse' : [rmse],
            'mse' : 'mse',
            'val_loss' : 'val_loss',
            'mae' : 'mae'
            }
        if self.existing_model == False:
            model = model_dict[settings['model']]

        elif self.existing_model == True:
            model = load_model(settings)
        else:
            raise Error
        optimizer = keras.optimizers.Adam(
            learning_rate = settings['learning_rate'],
            beta_1 = 0.9,
            beta_2 = 0.999,
            epsilon = 1e-07,
            amsgrad = False)
        model.compile(
            optimizer = optimizer, 
            loss = settings['loss'],
            metrics = metric_dict[settings['metric']])
        fig = plot_model(
            model, 
            to_file = settings['model_plot_path'] + settings['model'] + '.png',
            show_shapes = False,
            #show_dtype = True,
            show_layer_names = False)
        model.summary()
        self.model = model
        #self.score = None
        #self.loss = [None]

    def train(self, data):
        tic = time.time()
        self.history = [None]
        self.loss = [None]
        self.val_loss = [None]
        keys = list(series_stacks.keys())
        for h in range(len(keys)):
            series_stack = series_stacks[keys[h]]
            #print('\nTraining on ', keys[h],'% healthy data.\n')
            #print('\n Number of series being used for training:', int(len(series_stack[self.settings['preprocess_type']])*(self.settings['data_split']['train']+self.settings['data_split']['validation'])/100), '\n')
            X, Y = data_splitter(self, series_stack, ['train', 'validation'])
            if self.settings['noise'] == True:
                X += (np.random.rand(np.shape(X)[0],np.shape(X)[1],np.shape(X)[2])-self.settings['noise_mean'])/self.settings['noise_var']
            if np.shape(X)[0] == 0:
                pass
            else:         
              history = self.model.fit(
                  x = X,#patterns,
                  y = Y,#targets, 
                  batch_size = self.settings['batch_size'],
                  epochs=self.settings['epochs'], 
                  verbose=1,
                  callbacks=self.early_stopping, #self.learning_rate_scheduler],
                  validation_split = self.settings['data_split']['validation']/100,
                  shuffle = self.settings['shuffle'])
              self.history.append(history)
              self.loss.extend(history.history['loss'])
              self.val_loss.extend(history.history['val_loss'])  
              if self.settings['save_periodically'] == True and i % self.settings['save_interval'] == 0:
                  save_model(self.model,self.settings)  
        self.model.summary()
        self.toc = np.round(time.time()-tic,1)
        print('Elapsed time: ', self.toc)
        return

