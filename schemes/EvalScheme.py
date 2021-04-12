from Model import *
from SQLAConnection import SQLAConnection 
from QueryGenerator import QueryGenerator
from Data import *


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
        model.compile_model()

        
        connection = SQLAConnection()
        query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.start_date,
            self.settings.end_date
            )
        # Instansiera data
        data = NewData(query_generator,connection)
        data.figure_out_length(model)
        data.make_new_df_postgres()
        data.find_discontinuities()
        data.split_at_discontinuities()
        data.preprocess(self.settings.normalization)
        data.train_test_split(self.data_split)      
        model.make_timeseries_dataset(data)
        #print(data.dfs)
        # Läs in neuralnät
        model.load_nn()
        model.test()
        model.plot_outliers()
        model.plot_example()
        
