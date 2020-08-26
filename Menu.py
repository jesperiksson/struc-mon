
def menu():
    while True: # The infinite program-loop
        action = prompt_action()
        if action == 'quit':
            quit()
        elif action == 'new model':
            model = new_model()
        elif action == 'load_model':
            model = load_model()
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
    return input('What to do?')  

def quit():
    quit()







