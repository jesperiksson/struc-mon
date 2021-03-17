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
        train_query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.start_date,
            self.settings.end_date
            )
        train_data = Data(train_query_generator,connection)
        test_query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.end_date,
            self.settings.test_end_date
            )
        test_data = Data(train_query_generator,connection)
        train_data.make_df_postgres()
        train_data.preprocess(self.settings.normalization)
        test_data.make_df_postgres()
        test_data.preprocess(self.settings.normalization)
        train_data.add_trig()
        test_data.add_trig()
        model.setup_nn()
        model.data_split.train = 0.9
        model.data_split.validation = 0.1
        model.data_split.test = 0
        model.train_test_split(train_data.df)
        model.make_timeseries_dataset()
        if self.args.load: # Load a neural, either with name from settings or with the name the user provided if it provided
            model.load_nn()

        model.train()
        #model.print_summary()
        model.save_nn(overwrite=True)
        #model.plot_history()
        #model.evaluate()
        
        model.data_split.train = 0
        model.data_split.validation = 0
        model.data_split.test = 1
        model.train_test_split(test_data.df)
        model.make_timeseries_dataset()
        model.test()
        model.plot_outliers()
        model.plot_example()
