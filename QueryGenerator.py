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

    def generate_where(self):
        where_clause = f"{config.schema}.{config.table_names[self.sensors[0]]}.ts BETWEEN \'{self.parse_date(self.start_date)}\' AND \'{self.parse_date(self.end_date)}\' "
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
        query += f"WHERE {self.generate_where()}"
        query += self.generate_and()
        return query

