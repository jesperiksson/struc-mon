from datetime import datetime, timedelta, date

from Model import *
from SQLAConnection import SQLAConnection 
from QueryGenerator import QueryGenerator
from ReportGenerator import ReportGenerator
from Data import *
from PostgresData import *
from LinkGenerator import LinkGenerator
from PCAAnomalies import *
from AnomalySettings import *
import config

class Scheme():
    def __init__(self,args, settings, data_split):
        self.args = args
        self.settings = settings
        self.data_split = data_split
    

    def execute_scheme(self):
        #model = TimeSeriesClassificationNeuralNet(self.settings)
        model = TimeSeriesPredictionNeuralNet(self.settings)
        connection = SQLAConnection()
        query_generator = QueryGenerator(
            self.settings.sensors,
            self.settings.start_date,
            self.settings.end_date
            )
        report_generator = ReportGenerator(self.settings)
        link_generator = LinkGenerator(self.settings)
        data = AnomalyData(link_generator,connection)
        #data.generate_metadata_report(ReportGenerator(self.settings))
        #data.make_df()
        #data.save_df(name=self.settings.dataset_name)
        
        #data.find_discontinuities()
        #data.split_at_discontinuities()
        #data.plot_data()
        #data.add_temp()
        #data.save_dfs(name=self.settings.dataset_name)
        #data.load_dfs(date='2020-11-01')
        #data.load_extend_dfs(date='2020-11-13')
        startdate = datetime.strptime('2020-11-20',config.dateformat)
        data.load_dfs(date=datetime.strftime(startdate,config.dateformat))
        dates_ahead = 2
        mode = 'while'
        if mode == 'for':
            for i in range(dates_ahead):
            
                data.load_extend_dfs(date=datetime.strftime(startdate+timedelta(days=i), config.dateformat))
              
        elif mode == 'while': 
            tdate = startdate      
            while tdate.date() != date.today():
                try:
                    data.load_extend_dfs(date=datetime.strftime(tdate, config.dateformat))
                    
                except FileNotFoundError:
                    pass
                tdate = tdate+timedelta(days=1)
        data.purge_empty_dfs()  
        data.preprocess()
        data.merge_dfs()
        #data.find_correlation()
        anomaly_settings = AnomalySettings()
        data.filter_hours('02:00:00','05:00:00')
        data.set_anomaly_settings(anomaly_settings)
        for feature in anomaly_settings.anomaly_sensor:
            #data.locate_anomalies_filtered_dfs(feature)
            data.locate_anomalies_dfs(feature)
            #data.save_plots(feature)
            #data.plot_filtered_hours(foi = feature)
        #data.plot_filtered_hours(foi = 'acc1_ch_z')#,project_anomalies = 'acc1_ch_z')
        pca = PCAAnomalies(data.anomalies)
        pca.fit_PCA()
        #pca.get_cov()
        #anomaly_key, df_number = pca.get_argmax(col='sigma')
        pca.plot_components()
        pca.scree_plot()
        pca.plot_hist()
        pca.plot_components_3d()
        #pca.plot_components()
        #data.plot_anomalies(df_num = df_number, anomaly_key = anomaly_key)
        #data.ssa()
        #data.add_trig()
        #data.train_test_split(self.data_split)
        #data.save_dfs(name = f"{self.settings.dataset_name}_split")
        #model.setup()
        #model.compile_model()
        #model.make_timeseries_dataset(data)
        #model.print_shape()
        #model.load_dataset()
        #model.inspect_dataset()
        #model.train()
        #model.save_dataset()
        #model.plot_history()
        #model.evaluate()
        #model.save_nn()
        #model.test()
        #model.plot_outliers()
        #model.plot_example()
        
        
        

