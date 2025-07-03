import sys
import argparse
import sqlalchemy
import pandas as pd
import os
import importlib.util

SQL_FILE = 'src/sql/congestion_results.sql'

# Dynamically import db_config from src/modules
MODULE_PATH = os.path.join(os.path.dirname(__file__), 'src', 'modules', 'db_config.py')
spec = importlib.util.spec_from_file_location('db_config', MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load db_config from {MODULE_PATH}")
db_config = importlib.util.module_from_spec(spec)
sys.modules['db_config'] = db_config
spec.loader.exec_module(db_config)
engine = db_config.engine

def main(run_configs_id, iteration_id):
    # Read SQL file
    with open(SQL_FILE, 'r') as f:
        sql = f.read()
    # Replace hardcoded IDs with parameters
    sql = sql.replace('run_configs_id  =17', f'run_configs_id  ={run_configs_id}')
    sql = sql.replace('run_config_id  =17', f'run_config_id  ={run_configs_id}')
    sql = sql.replace('iteration_id  =5', f'iteration_id  ={iteration_id}')
    sql = sql.replace('iteration_id = 5', f'iteration_id = {iteration_id}')
    # Split queries (naive split on ';')
    queries = [q.strip() for q in sql.split(';') if q.strip()]
    with engine.connect() as conn:
        for i, query in enumerate(queries):
            print(f'--- Query {i+1} ---')
            try:
                df = pd.read_sql(query, conn)
                print(df)
            except Exception as e:
                print(f'Error running query {i+1}: {e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run congestion_results.sql with parameters.')
    parser.add_argument('--run_configs_id', type=int, required=True)
    parser.add_argument('--iteration_id', type=int, required=True)
    args = parser.parse_args()
    main(args.run_configs_id, args.iteration_id) 