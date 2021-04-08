from Model import *
from SQLAConnection import SQLAConnection 
from QueryGenerator import QueryGenerator
from ReportGenerator import ReportGenerator
from Data import Data


class Scheme():
    def __init__(self,args, settings, data_split):
        self.args = args
        self.settings = settings
        self.data_split = data_split
    

    def execute_scheme(self):
        model = TimeSeriesClassificationNeuralNet(self.settings)
        connection = SQLAConnection()
        query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.start_date,
            self.settings.end_date
            )
        report_generator = ReportGenerator(self.settings)
        data = Data(query_generator,connection)
        #data.generate_metadata_report(ReportGenerator(self.settings))
        #data.load_df(name=self.settings.dataset_name)
        #data.make_df_postgres()
        #data.save_df(name=self.settings.dataset_name)
        
        #data.find_discontinuities()
        #data.split_at_discontinuities()
        #data.plot_data()
        #data.add_temp()
        #data.save_dfs(name=self.settings.dataset_name)
        data.load_dfs(date='2020-11-01')

        #data.load_extend_dfs(date='2020-12-02')
        #data.preprocess(self.settings.normalization)
        #data.add_trig()
        data.train_test_split(self.data_split)
        #data.save_dfs(name = f"{self.settings.dataset_name}_split")
        model.setup()
        model.make_timeseries_category_dataset(data)
        model.print_shape()
        #model.load_dataset()
        #model.inspect_dataset()
        model.train()
        #model.save_dataset()
        model.plot_history()
        model.evaluate()
        model.save_nn()
        model.test()
        model.plot_outliers()
        #model.plot_example()
        
        
        

