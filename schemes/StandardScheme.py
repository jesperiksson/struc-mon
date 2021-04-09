from Model import *
from SQLAConnection import SQLAConnection 
from QueryGenerator import QueryGenerator
from Data import Data


class Scheme():
    def __init__(self,args, settings, data_split):
        self.args = args
        self.settings = settings
        self.data_split = data_split
    

    def execute_scheme(self):
        # Instansiera modell
        model = TimeSeriesPredictionNeuralNet(self.settings)
        # Importera modellinställningar
        model.setup()
        # Läs in neuralnät
        #model.load_nn()
        
        connection = SQLAConnection()
        query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.start_date,
            self.settings.end_date
            )
        # Instansiera data
        
        data = Data(query_generator,connection)
        '''
        data.figure_out_length(model)
        data.make_new_df_postgres()
        data.preprocess(self.settings.normalization)
        data.train_test_split(self.data_split)      
        model.make_timeseries_dataset(data)
        model.test()

        
        '''
        data.make_df_postgres()
        data.find_discontinuities()
        data.split_at_discontinuities()
        data.preprocess(self.settings.normalization)
        data.add_trig()
        data.train_test_split(self.data_split)      
        model.make_timeseries_dataset(data)
        model.print_shape()
        model.plot_example()
            
           
        model.train()
        model.plot_history()
        model.evaluate()
        model.save_nn(overwrite=True)
        model.test()
        model.plot_outliers()
        model.plot_example()

