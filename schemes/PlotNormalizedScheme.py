from Model import TimeSeriesNeuralNet
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
        model = TimeSeriesNeuralNet(self.settings)
        connection = SQLAConnection()
        query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.start_date,
            self.settings.end_date
            )
        report_generator = ReportGenerator(self.settings)
        data = Data(query_generator,connection)
        data.make_df_postgres()
        data.find_discontinuities()
        data.split_at_discontinuities()      
        data.preprocess(self.settings.normalization)
        data.plot_normalized()
