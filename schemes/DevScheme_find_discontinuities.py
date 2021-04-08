from Model import TimeSeriesNeuralNet, StatModel
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
        model = StatModel(self.settings)
        connection = SQLAConnection()
        query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.start_date,
            self.settings.end_date
            )
        report_generator = ReportGenerator(self.settings)
        data = Data(query_generator,connection)
        #data.generate_metadata_report(ReportGenerator(self.settings))
        data.make_df_postgres()
        data.find_discontinuities()
        #data.preprocess(self.settings.normalization)
        #data.plot_data()
        #data.add_trig()
        #data.add_temp()
        #data.train_test_split(self.data_split)
        #model.setup_model()
        #model.make_timeseries_dataset(data,print_shape=True)
        

