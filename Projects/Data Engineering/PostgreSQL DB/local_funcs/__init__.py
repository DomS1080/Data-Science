import sqlalchemy as sa
import pandas as pd

query_db_names = "SELECT datname FROM pg_database WHERE datistemplate = false;"

def print_dbs_df(engine):
    """Query and output server dbs as pandas dataframe"""
    with sa.future.Connection(engine) as conn:
        result = conn.execute(sa.text(query_db_names))
        assert type(result) == sa.engine.cursor.CursorResult
        result = [*result]
        print("Databases:", pd.DataFrame(result), sep='\n')


def print_dbs_list(engine):
    """Query and output server dbs as list"""
    with sa.future.Connection(engine) as conn:
        result = conn.execute(sa.text(query_db_names))
        assert type(result) == sa.engine.cursor.CursorResult
        result = [*result]
        print("Databases:", [x[0] for x in result])