from Model import TimeSeriesNeuralNet
from SQLAConnection import SQLAConnection 
from QueryGenerator import QueryGenerator
from Data import Data


class Scheme():
    def __init__(self,args, settings):
        self.args = args
        self.settings = settings
    

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
        df = data.df
        model.setup_nn()
        model.train_test_split(df)
        model.make_timeseries_dataset()
        if self.args.load: # Load a neural, either with name from settings or with the name the user provided if it provided
            model.load_nn()
            
        model.train()
        model.print_summary()
        model.save_nn(overwrite=True)
        model.plot_history()
        model.evaluate()
        model.test()
        model.plot_outliers()
        model.plot_example()
