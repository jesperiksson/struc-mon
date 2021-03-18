from Model import TimeSeriesNeuralNet
from SQLAConnection import SQLAConnection 
from QueryGenerator import QueryGenerator
from Data import Data, NewData


class Scheme():
    def __init__(self,args, settings, data_split):
        self.args = args
        self.settings = settings
        self.data_split = data_split
        self.data_split.train = 0.2
        self.data_split.validation = 0.4
        self.data_split.test = 0.4

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
        model.make_timeseries_dataset(data,print_shape=True)
        model.load_nn()
        model.evaluate()    
        model.train_classifier_parameters()
        model.set_up_classifier()
        new_data = NewData(query_generator,connection)
        new_data.figure_out_length(model)
        new_data.make_new_df_postgres()
        new_data.preprocess(self.settings.normalization)
        new_data.add_trig()
        print(new_data.df)
        self.data_split.train = 0
        self.data_split.validation = 0
        self.data_split.test = 1
        new_data.train_test_split(self.data_split)
        model.make_timeseries_dataset(new_data,print_shape=False)
        model.classify()
        #model.plot_outliers()
        #model.plot_example()
            
            
            
        
