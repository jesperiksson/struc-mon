import datetime

import config

class QueryGenerator():
    def __init__(self,sensors, start_date, end_date):
        self.sensors = sensors
        self.start_date = start_date
        self.end_date = end_date
        
        
    def generate_select(self):
        select_command = ''
        for sensor in self.sensors:
            table = config.table_names[sensor]
            for column in config.column_names[table]:
                select_command+=table+'.'+column+' AS '+sensor+'_'+column+' ,' 
        select_command += f"{config.table_names[self.sensors[0]]}.ts AS ts "
        return select_command

    def generate_where_id(self):
        where_clause = f"{config.schema}.{config.table_names[self.sensors[0]]}.id < {self.settings.n_samples}"
        return where_clause

    def generate_where(self,table_name=config.table_names['acc1']):
        where_clause = f"{config.schema}.{table_name}.ts BETWEEN \'{self.parse_date(self.start_date)}\' AND \'{self.parse_date(self.end_date)}\' "
        return where_clause

    def generate_and(self):
        and_clause = ''
        if len(self.sensors)>1:
            for i in range(len(self.sensors)-1):
                and_clause += f"AND {config.schema}.{config.table_names[self.sensors[0]]}.ts = {config.schema}.{config.table_names[self.sensors[i+1]]}.ts "
                # ts ensures integrity in data
        return and_clause

    def parse_date(self,date):
        
        return str(datetime.datetime.strptime(date,config.dateformat))

    def generate_query(self):
        query = ''
        query += f"SELECT {self.generate_select()}"
        query += f"FROM {config.schema}.{(', '+config.schema+'.').join([config.table_names[sensor] for sensor in self.sensors])} "
        query += f"WHERE {self.generate_where(table_name=config.table_names[self.sensors[0]])} "
        query += f"ORDER BY {config.schema}.{config.table_names[self.sensors[0]]}.ts ASC"
        query += self.generate_and()
        return query
        
    def generate_temp_query(self):
        query = ''
        query += f"SELECT {config.table_names['temp']}.{''.join(config.column_names[config.table_names['temp']])} AS temp "
        query += f"FROM {config.schema}.{config.table_names['temp']} "
        query += f"WHERE {self.generate_where(table_name = config.table_names['temp'])}"
        return query
        
    def fetch_latest_query(self,model): # Needs a model object to figure out how many tuples to request
        query = ''
        return query
        
    def generate_latest_query(self,steps=50): # Needs a model object to figure out how many tuples to request
        query = ''
        query += f" SELECT {self.generate_select()}"
        query += f" FROM {config.schema}.{(', '+config.schema+'.').join([config.table_names[sensor] for sensor in self.sensors])} "
        query += f" ORDER BY id DESC LIMIT {steps} "
        return query
        
    def generate_metadata(self):
        query = ''
        query += f"SELECT ts "
        query += f"FROM {config.schema}.{config.table_names[self.sensors[0]]} "
        return query
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

