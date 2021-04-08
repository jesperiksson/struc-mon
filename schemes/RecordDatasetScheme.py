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
        model = TimeSeriesNeuralNet(self.settings)
        connection = SQLAConnection()
        query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.start_date,
            self.settings.end_date
            )
        report_generator = ReportGenerator(self.settings)
        data = Data(query_generator,connection)
        #data.generate_metadata_report(ReportGenerator(self.settings))
        #data.make_df_postgres()
        data.load_dfs('2020-11-01')
        data.load_extend_dfs(name='2020-11-02')
        data.load_extend_dfs(name='2020-11-03')
        #data.load_extend_dfs(name='2020-11-04')
        #data.load_extend_dfs(name='2020-11-05')
        #data.load_extend_dfs(name='2020-11-06')
        #data.load_extend_dfs(name='2020-11-07')
        #data.load_extend_dfs(name='2020-11-08')
        #data.load_extend_dfs(name='2020-11-09')
        #data.load_extend_dfs(name='2020-11-10')
        #data.load_extend_dfs(name='2020-11-11')
        #data.load_extend_dfs(name='2020-11-12')
        #data.load_extend_dfs(name='2020-11-13')
        #data.load_extend_dfs(name='2020-11-14')
        #data.load_extend_dfs(name='2020-11-15')
        '''
        data.load_extend_dfs(name='2020-11-16')
        data.load_extend_dfs(name='2020-11-17')
        data.load_extend_dfs(name='2020-11-18')
        data.load_extend_dfs(name='2020-11-19')
        data.load_extend_dfs(name='2020-11-20')
        data.load_extend_dfs(name='2020-11-21')
        data.load_extend_dfs(name='2020-11-22')
        data.load_extend_dfs(name='2020-11-23')
        data.load_extend_dfs(name='2020-11-24')
        data.load_extend_dfs(name='2020-11-25')
        data.load_extend_dfs(name='2020-11-26')
        '''
        #data.plot_data()
        #data.add_temp()
        data.train_test_split(self.data_split)
        model.setup()
        model.make_timeseries_dataset(data,print_shape=True)
        #model.save_dataset()
        model.train()
        model.save_nn(overwrite=True)
        model.plot_history()
        model.evaluate()
        model.test()
        model.plot_outliers()
        model.plot_example()
        

