# The top level program file
# Imports

# Libraries and modules
import argparse

# Other files
from Data import AccSeriesStack, InclSeriesStack, StrainSeriesStack
from Model import TimeSeriesNeuralNet
from Settings import Settings
import config
import use_cases


def main(): 
    # This parser allows the user to set flags 
    parser = argparse.ArgumentParser(
        description = '',
        epilog = '')
    parser.add_argument(
        '--q', 
        help = 'Dont show progress during training and evaluation',
        action = "store_true",
        #metavar = 'QUIET'
        )
    parser.add_argument(
        '--sensor',
        action = 'store',
        nargs = 1,
        default = 'acc',
        type = str,
        help = 'Which kind of data to use. These are available(keyword-sensor): \nacc - Acceleration[g] \nincl - Inclinations[deg] \nstrain - Strain[mV]'
        )
    parser.add_argument(
        '--preset',
        action = 'store',
        nargs = 1,
        default = None,
        type = str,
        help = 'Uses a different preset neural net'
        )
    parser.add_argument(
        '--load',
        action='store_true',
        #nargs = 0,
        help = 'If provided the model tries to load an existing neural net')
    parser.add_argument(
        '--name',
        action = 'store',
        nargs = 1,
        default = None,
        type = str,
        #choices = , TODO: list exixting models
        help='Try to load a model with the current name (given the --load argument is provided)')
    args = parser.parse_args()
    '''
    TBD: make a class for different run-configurations
    '''
    settings = Settings()
    if args.preset is not None:
        settings.preset = args.preset[0]
        print(settings.preset[0])
    if args.load and args.name != None: # Load an existing neural net with a given name    
        settings.name = args.name[0]
    print(settings.name)
    if args.q:
        model.settings_nn.verbose = 0   
    if args.sensor == 'incl':
        series_stack = InclSeriesStack(settings,'new',file_path = config.measurements)
        settings.sensor = 'incl'
    elif args.sensor == 'strain':
        series_stack = StrainSeriesStack(settings,'new',file_path = config.measurements)
        settings.sensor = 'strain'
    else: # Either acc by default or explicitly - doesnt matter
        series_stack = AccSeriesStack(settings,'new',file_path = config.measurements)
        settings.sensor = 'acc'
    series_stack.populate_stack()
    
    model = TimeSeriesNeuralNet(settings)
    learned = model.setup_nn()
    
    model.make_dataframe(series_stack)
    model.make_timeseries_dataset()
    if args.load: # Load a neural, either with name from settings or with the name the user provided if it provided
        model.load_nn()
    else:
        model.train()
        model.print_summary()
        model.save_nn(overwrite=True)
        model.plot_history()
    model.evaluate()
    model.test()
    model.plot_outliers()
    model.plot_example()
            

if __name__ == "__main__":
    main()
