# This file makes neural net modules from the command line by importing an appropriate template module
import importlib as il


def main():
    
        sys.path.append(config.model_path+self.name)
        module = il.import_module(self.settings.model) # Import the specified model from file with same name
if __name__ == "__main__":
    main()
