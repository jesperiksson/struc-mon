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
        model = TimeSeriesPredictionNeuralNet(self.settings)
        model.setup()
        connection = SQLAConnection()
        query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.start_date,
            self.settings.end_date
            )
        data = Data(query_generator,connection)
        if self.args.load_dataframe:
            data.load_dfs(date='2020-11-01')
            #data.load_extend_dfs(date='2020-11-02')
            #data.load_extend_dfs(date='2020-11-03')
            #data.load_extend_dfs(date='2020-11-04')
            #data.load_extend_dfs(date='2020-11-08')
            #data.load_extend_dfs(date='2020-11-12')
            data.load_extend_dfs(date='2020-11-16')
            #data.load_extend_dfs(date='2020-11-20')
            data.load_extend_dfs(date='2020-11-24')
            #data.load_extend_dfs(date='2020-11-28')
            data.load_extend_dfs(date='2020-12-02')  
            data.load_extend_dfs(date='2020-12-18')
            data.load_extend_dfs(date='2020-12-30')       
            data.load_extend_dfs(date='2020-12-08')      
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
        model.plot_example()
        #model.save_dataset()           
        if self.args.load: 
            model.load_nn()
           
        model.train()
        model.plot_history()
        model.evaluate()
        model.save_nn(overwrite=True)
        model.test()
        model.plot_outliers()
        model.plot_example()
