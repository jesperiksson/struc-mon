import importlib as il
import sys
import config
#from Data import SeriesStack
from Settings import DataSplit

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
            
        scheme_dict = {
            'train' : 'TrainScheme',
            'eval' : 'EvalScheme',
            'plot_normalized' : 'PlotNormalizedScheme',
            'sep_train_eval' : 'SepTrainTestScheme',
            'standard' : 'StandardScheme',
            'dev' : 'DevScheme',
            'record_dataset' : 'RecordDatasetScheme',
            'record_dataframes' : 'RecordDataframesScheme',
            'plot' : 'PlotScheme',
            'category' : 'CategoryScheme',
            'cont_eval' : 'ContinuosEvalScheme',
            'pcakmeans' : 'PCA_Kmeans_Scheme'
        }
        data_split = DataSplit()
        sys.path.append(config.scheme_path)
        scheme_module = il.import_module(scheme_dict[args.mode[0]])
        if args.date is not None:
            settings.start_date = args.date[0]
            settings.end_date = args.date[1]
            if len(args.date) == 3:
                settings.test_end_date = args.date[2]
        scheme = scheme_module.Scheme(args, settings, data_split)
        scheme.execute_scheme()
            

                
                
        
            


