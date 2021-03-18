from Model import TimeSeriesNeuralNet
from SQLAConnection import SQLAConnection 
from QueryGenerator import QueryGenerator
from Data import Data


class Scheme():
    def __init__(self,args, settings, data_split):
        self.args = args
        self.settings = settings
        self.data_split = data_split
    

    def execute_scheme(self):
        model = TimeSeriesNeuralNet(self.settings)
        connection = SQLAConnection()
        query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.start_date,
            self.settings.end_date
            )
        data = Data(query_generator,connection)
        data.make_df_postgres()
        data.preprocess(self.settings.normalization)
        data.add_trig()
        data.train_test_split(self.data_split)
        model.setup()
        model.make_timeseries_dataset(data)
        if self.args.load: # Load a neural, either with name from settings or with the name the user provided if it provided
            model.load_nn()
            
        model.train()
        #model.print_summary()
        model.save_nn(overwrite=True)
        #model.plot_history()
        #model.evaluate()
