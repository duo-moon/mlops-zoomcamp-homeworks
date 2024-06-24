import os
import pickle
from datetime import datetime
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric
from evidently.report import Report

from database import PostgresqlDatabase

load_dotenv('.env')

NUMERICAL_FEATURES = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
CATEGORICAL_FEATURES = ['PULocationID', 'DOLocationID']

db = PostgresqlDatabase(
    host='localhost',
    port=int(os.getenv('PG_PORT')),
    database=os.getenv('PG_DB'),
    user=os.getenv('PG_USER'),
    password=os.getenv('PG_PASSWORD'),
)

RAW_DATA = pd.read_parquet('data/green_tripdata_2024-03.parquet')
REFERENCE_DATA = pd.read_parquet('data/05_monitoring_reference.parquet')

with open('../output/models/lr_05_monitoring.bin', 'rb') as f:
    MODEL = pickle.load(f)


def init_db() -> None:
    query = """
        CREATE TABLE IF NOT EXISTS evidently_metrics (
        	event_time              TIMESTAMP,
            prediction_drift        FLOAT,
            num_drifted_columns     INTEGER,
	        share_missing_values    FLOAT,
	        fare_amount_quantile    FLOAT
	    )
    """
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()


def create_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, event_datetime: datetime) -> tuple:
    column_mapping = ColumnMapping(
        prediction='prediction',
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        target=None
    )
    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name='fare_amount', quantile=0.5),
    ])

    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    result = report.as_dict()
    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    fare_amount_quantile = result['metrics'][3]['result']['current']['value']

    return event_datetime, prediction_drift, num_drifted_columns, share_missing_values, fare_amount_quantile


def calculate_metrics() -> list[tuple]:
    results = []
    for i in range(1, 30):
        current_data = (
            RAW_DATA
            .loc[lambda df: (df['lpep_pickup_datetime'] >= datetime(2024, 3, i)) & (
                    df['lpep_pickup_datetime'] < datetime(2024, 3, 1) + timedelta(days=i))]
        )
        current_data['prediction'] = MODEL.predict(current_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].fillna(0))
        results.append(
            create_report(
                reference_data=REFERENCE_DATA,
                current_data=current_data,
                event_datetime=datetime(2024, 3, i, 2, 0, 0),
            )
        )

    return results


def insert_data(data: list[tuple]) -> None:
    query = """
        INSERT INTO evidently_metrics (event_time, prediction_drift, num_drifted_columns, 
                                       share_missing_values, fare_amount_quantile)
        VALUES(%s,%s,%s,%s, %s)
    """
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, data)
            conn.commit()


def generate_metrics() -> None:
    init_db()
    results = calculate_metrics()
    insert_data(data=results)


if __name__ == '__main__':
    generate_metrics()
