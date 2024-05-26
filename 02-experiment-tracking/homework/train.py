import os
import pickle

import click
import mlflow
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

load_dotenv('.env')
MLFLOW_HOST = os.getenv('MLFLOW_HOST')
MLFLOW_PORT = os.getenv('MLFLOW_PORT')
MLFLOW_USER = os.getenv('MLFLOW_USER')
MLFLOW_PASSWORD = os.getenv('MLFLOW_PASSWORD')

mlflow.set_tracking_uri(uri=f'http://{MLFLOW_USER}:{MLFLOW_PASSWORD}@{MLFLOW_HOST}:{MLFLOW_PORT}')
mlflow.set_experiment('random-forest-model-train')
mlflow.autolog()


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric('val_rmse', rmse)


if __name__ == '__main__':
    run_train()
