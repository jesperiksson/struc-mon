import datetime
class ReportGenerator():
    def __init__(self,settings):
        self.settings = settings
        
    def generate_training_report(self,model):
        dateformat = '%Y-%m-%d %H:%M:%S'
        training_report = f"##############\n"
        training_report += f"{datetime.datetime.now().strftime(dateformat)}\n"
        training_report += f"Train dataset start: {model.train_df['ts'].iloc[0]}\n" 
        training_report += f"Train dataset end: {model.train_df['ts'].iloc[-1]}\n"
        training_report += f"batch size: {model.settings_train.batch_size}\n"
        training_report += f"epochs: {model.settings_train.epochs}\n"
        training_report += f"##############\n"
        return training_report
