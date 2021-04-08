from Model import *
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
        model = TimeSeriesPredictionNeuralNet(self.settings)
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
            if self.args.load_dataframe:
                data.load_dfs(date='2020-11-01')
                data.load_extend_dfs(date='2020-11-17')
                data.load_extend_dfs(date='2020-11-25')
                data.load_extend_dfs(date='2020-12-03')  
                data.load_extend_dfs(date='2020-12-17')
                data.load_extend_dfs(date='2020-12-29')       
                data.load_extend_dfs(date='2020-12-09')      
            else:
                data.make_df_postgres()
                data.find_discontinuities()
                data.split_at_discontinuities()
                data.preprocess(self.settings.normalization)
                #data.fast_fourier_transform()
                #data.wawelet()
                #data.STL()
                data.add_trig()
                #data.add_temp()
            data.train_test_split(self.data_split)      
            model.make_timeseries_dataset(data)
            model.print_shape()
            #model.plot_example()
            #model.save_dataset()  
        model.load_nn()
        model.test()
        model.plot_outliers()
        model.plot_example()        
            
            
            
        
