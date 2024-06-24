import os
import random
import uuid
from datetime import datetime
from datetime import timedelta

from dotenv import load_dotenv

from database import PostgresqlDatabase

load_dotenv('.env')

db = PostgresqlDatabase(
    host='localhost',
    port=int(os.getenv('PG_PORT')),
    database=os.getenv('PG_DB'),
    user=os.getenv('PG_USER'),
    password=os.getenv('PG_PASSWORD'),
)


def init_db() -> None:
    query = """
        CREATE TABLE IF NOT EXISTS dummy_metrics (
        	event_time  TIMESTAMP,
            value1      INTEGER,
            value2      VARCHAR,
	        value3      FLOAT
	    )
    """
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()


def insert_dummy_data() -> None:
    data = [
        (datetime.now() + timedelta(seconds=i), random.randint(0, 1000), str(uuid.uuid4()), random.random())
        for i in range(100)
    ]
    query = """INSERT INTO dummy_metrics VALUES(%s,%s,%s,%s)"""

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, data)
            conn.commit()


def generate_dummy_metrics() -> None:
    init_db()
    insert_dummy_data()


if __name__ == '__main__':
    generate_dummy_metrics()
