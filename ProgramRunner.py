from Data import SeriesStack
from Model import TimeSeriesNeuralNet

class ProgramRunner():
    def __init__(self,settings,args):

        if args.preset is not None:
            settings.preset = args.preset[0]
        if args.load and args.name != None: # Load an existing neural net with a given name    
            settings.name = args.name[0]
            
        if args.q: # Silent by default
            model.settings_nn.verbose = 0  
        if 'incl' in args.sensor:
            settings.sensors.append('incl')
        elif 'strain' in args.sensor:
            settings.sensors.append('strain')
        elif 'acc1' in args.sensor:
            settings.sensors.append('acc1')
        elif 'acc2' in args.sensor:
            settings.sensors.append('acc2')
        else: # Either acc by default or explicitly - doesnt matter
            pass # use setting from Settings

        series_stack = SeriesStack(settings)
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
