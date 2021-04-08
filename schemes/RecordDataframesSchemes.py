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
        model = NeuralNet(self.settings)
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
        data.make_df_postgres()
        #data.save_df(name=self.settings.dataset_name)
        
        data.find_discontinuities()
        data.split_at_discontinuities()
        data.preprocess(self.settings.normalization)
        #data.plot_data()
        data.add_trig()
        #data.add_temp()
        data.save_dfs(name=self.settings.start_date)
        #data.load_dfs(name=self.settings.dataset_name)
        #data.train_test_split(self.data_split)
        #model.setup()
        #model.make_timeseries_dataset(data,print_shape=True)
        #model.save_dataset()
        
        

