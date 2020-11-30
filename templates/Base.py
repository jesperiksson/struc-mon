class Base():
    def __init__(self):
        pass
        
    def make_choice(self,prompt,default=None):
        value = input(prompt+', Default: '+default)
        if default != None and value == None:
            value = default
        return value
        
    def make_multiple_choices(self): # TODO
        pass
