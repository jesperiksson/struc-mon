
def check_if_existing():

    return


### FUNCTIONS FOR CREATING OR LOADING A MODEL ###
def new_model():
    settings = set_settings() # get_settings() lives in a separate file since it is large
    model = NeuralNet(settings, existing_model = False)
    model.settings['learned'] = {}
    read_and_train(model)
    return model

def load_model():
    settings = load_settings(input('Which model?'))
    model = NeuralNet(settings, existing_model = True)
    read_and_train(model)
    return model

def read_and_train(model)
    if model.settings['data_function'] == 'one_by_one':
        to_learn = scan(learned = model.settings['learned'])
        for i in range(len(to_learn)):
            data, learned = get_data_one_by_one(to_learn[i])
            model.settings.update({'learned' : learned})
            NeuralNet.train(model, data)
    elif model.settings['data_function'] == 'all_at_once':
        data = get_data_all_at_once(learned = {})
        model.settings.update({'learned' : learned})
        NeuralNet.train(model, data)
    save_model(model)

def get_data_one_by_one(to_learn):
    data = Databatch(TBIFUNC(to_learn))  
    return data, learned

def get_data_all_at_once(to_learn):
    data = Databatch(TBIFUNC(to_learn))
    return data, learned

def load_model():
    settings = load_settings()
    model = NeuralNet(settings, existing_model = True)

def load_settings():
    settings = None
    return settings

def save_architecture(arch):
    with open(arch['model_path']+ arch['fname'] +'.pkl', 'wb') as f:
        pickle.dump(arch, f, pickle.HIGHEST_PROTOCOL)

def load_architecture(arch):
    with open(arch['model_path']+ arch['fname'] +'.pkl', 'rb') as f:
        return pickle.load(f)

def scan(learned):
    available = None # TBI once file structure is known
    to_learn = available - learned
    return to_learn

### FUNCTIONS FOR CONTINUOSLY UPDATING ###

def continuosly_update(model):
    # Start the scheduler
    scheduler = BackgroundScheduler()
    #scheduler.daemonic = False
    scheduler.start()
    scheduler.add_job(check_for_new_files(model), 'interval', minutes = interval) 
    return

def check_for_new_files(model)
    to_learn = scan(model.settings[learned])
    if model.settings['data_function'] == 'one_by_one':
        for i in range(len(to_learn)):
            data, learned = get_data_one_by_one(to_learn[i])
            model.settings.update({'learned' : learned})
            NeuralNet.train(model, data)
    elif model.settings['data_function'] == 'all_at_once':
        data = get_data_all_at_once(learned = {})
        model.settings.update({'learned' : learned})
        NeuralNet.train(model, data)
    save_model(model)
    return

### FUNCTIONS FOR ANALYZING ###

def analyze(model):
    pass
