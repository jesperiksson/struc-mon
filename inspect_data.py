# Imports

# Libraries and modules
import argparse

# Other files
from Data import AccSeriesStack, InclSeriesStack, StrainSeriesStack
from Settings import Settings
import config

def main():

    parser = argparse.ArgumentParser(
        description = '',
        epilog = '')
    parser.add_argument(
        '--sensor',
        action = 'store',
        nargs = 1,
        default = ['acc'],
        type = str,
        help = 'Which kind of data to use. These are available(keyword-sensor): acc - Acceleration[g], incl - Inclinations[deg], strain - Strain[mV]'
        )  
    parser.add_argument(
        '--start',
        action = 'store',
        nargs  = 1,
        default = [0],
        type = int,
        help = 'Index to start the plot at'
        ) 
    parser.add_argument(
        '--stop',
        action = 'store',
        nargs  = 1,
        default = [500],
        type = int,
        help = 'Index to stop the plot at'
        )      
    args = parser.parse_args()

    print(args.sensor)
    settings = Settings()
    
    if args.sensor[0] == 'incl':
        settings.sensor = 'incl'
        series_stack = InclSeriesStack(settings,'new',config.measurements)
        
        features = config.incl_features
    elif args.sensor[0] == 'strain':
        settings.sensor = 'strain'
        series_stack = StrainSeriesStack(settings,'new',config.measurements)
        
        features = config.strain_features
    else: # Either acc by default or explicitly - doesnt matter
        settings.sensor = 'acc'
        series_stack = AccSeriesStack(settings,'new',config.measurements)
        
        features = config.acc_features
        #
    series_stack.populate_stack()
    
    series = series_stack.pick_series()
    series.plot_data(features,settings.sensor,args.start[0],args.stop[0])
    series.filter_data(features)
    series.plot_filtered_data(features,settings.sensor,args.start[0],args.stop[0])
    series.rainflow(features,settings.sensor)
    

if __name__ == "__main__":
    main()
