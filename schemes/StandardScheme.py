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
        model.setup()
        if self.args.load_dataset:
            model.load_dataset()
        else:
            connection = SQLAConnection()
            query_generator = QueryGenerator(
                self.settings.sensors,
                self.settings.start_date,
                self.settings.end_date
                )
            data = Data(query_generator,connection)
            data.make_df_postgres()
            data.find_discontinuities()
            data.split_at_discontinuities()
            data.preprocess(self.settings.normalization)
            data.add_trig()
            #data.add_temp()
            data.train_test_split(self.data_split)      
            model.make_timeseries_dataset(data,print_shape=True)           
        if self.args.load: 
            model.load_nn()
           
        model.train()
        model.save_nn(overwrite=True)
        model.plot_history()
        model.evaluate()
        model.test()
        model.plot_outliers()
        model.plot_example()
