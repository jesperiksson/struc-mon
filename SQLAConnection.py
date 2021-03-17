import config
from sqlalchemy import create_engine
class SQLAConnection():
    def __init__(self):
        self.endpoint = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
            config.SQLA_GOST_DATABASE_USER,
            config.SQLA_GOST_DATABASE_PASS,
            config.SQLA_GOST_DATABASE_HOST,
            config.SQLA_GOST_DATABASE_PORT,
            config.SQLA_GOST_DATABASE_NAME,
        )
        #try:
        self.engine = create_engine(self.endpoint)
        
