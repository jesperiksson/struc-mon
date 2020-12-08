
# Standard packages
import sys
import argparse
# Self made modules
#from functions import *



class Menu():
    def __init__(self): 
        self.state = None
        self.model = 'None'
        parser = argparse.ArgumentParser()

        args = parser.parse_args()
        
    def __repr__(self):
        return repr(
            f'Current model: {self.model}')        

'''
def menu(action = None):
    while True: # The infinite program-loop
        if action == None:
            action = prompt_action()
        else:
            pass
        if action == 'quit':
            quit_program()
        elif action == 'new model':
            settings, data, model = new_model()
            model.make_dataframe()
            model.setup_nn()
            model.make_timeseries_dataset(mod)
            model.train()
            model.evaluate
            
        elif action == 'load_model':
            settings, data, neural_net = load_model()
        elif action == 'continously_update':
            try:
                continously_update(model)
            except NameError:
                print('No model')
                pass
        elif action == 'analyze':
            try:
                analyze()
            except NameError:
                print('No model')
                pass

def prompt_action():
    return input('\nWhat to do?\n')  

def quit_program():
    sys.exit(1)
'''






