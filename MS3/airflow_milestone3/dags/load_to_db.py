import pandas as pd
from sqlalchemy import create_engine

def load_to_postgres(path, lookup_path): 
    df = pd.read_csv(path)
    lookup = pd.read_csv(lookup_path)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/accidents_etl')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'UK_Accidents_1981',con = engine,if_exists='replace')
    lookup.to_sql(name = 'lookup_table',con = engine,if_exists='replace')