import datetime
import pandas as pd
class ReportGenerator():
    def __init__(self,settings):
        self.settings = settings
        self.dateformat = '%Y-%m-%d %H:%M:%S'
        
    def generate_training_report(self,model):
        
        training_report = f"##############\n"
        training_report += f"{datetime.datetime.now().strftime(self.dateformat)}\n"
        training_report += f"Train dataset dates: {model.dates}\n" 
        training_report += f"batch size: {model.settings_train.batch_size}\n"
        training_report += f"epochs: {model.settings_train.epochs}\n"
        training_report += f"##############\n"
        return training_report
        
    def generate_metadata_report(self,df):
        metadata = f"##############\n"
        metadata += f"{df.groupby(df.index.date).count()}"
        metadata += f"##############\n"
        return metadata
        
    def generate_eval_report(self,model):
        eval_report = f"##############\n"
        eval_report += f"{datetime.datetime.now().strftime(self.dateformat)}\n"
        eval_report += f"Test dataset dates: {model.dates}\n"
        eval_report += f"Scores: {[s for s, score in zip(model.nn.metrics_names, model.loss)]}\n"  
        eval_report += f"##############\n"
        return eval_report      
        
