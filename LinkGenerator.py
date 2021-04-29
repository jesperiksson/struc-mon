import config as c
class LinkGenerator():
    def __init__(self,settings):
        self.settings = settings
        
    def generate_JSON_link(self):
        link = c.OGC_datastreams
        link += f"({c.datastream_minute_sensor_id[self.settings.agg_sensor]})/"
        link += f"Observations"
        return link 
        
        
