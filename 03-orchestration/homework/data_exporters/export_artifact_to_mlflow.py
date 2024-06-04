import pathlib
import pickle

import mlflow
import mlflow.sklearn
from mage_ai.data_preparation.shared.secrets import get_secret_value
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

get_secret_value('<secret_name>')

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export(output: tuple[LinearRegression, BaseEstimator], **kwargs) -> None:
    output_path = pathlib.Path(kwargs['configuration'].get('output_path'))
    mlflow_user = get_secret_value('mlflow_user')
    mlflow_password = get_secret_value('mlflow_password')

    mlflow_uri = f'http://{mlflow_user}:{mlflow_password}@mlflow-server:5000'
    mlflow.set_tracking_uri(mlflow_uri)

    model, dv = output

    with open(output_path.joinpath('dv.pkl'), 'wb') as f:
        pickle.dump(dv, f)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            model,
            artifact_path='sklearn-models',
            registered_model_name='yellow-tripdata-lr-model'
        )
        mlflow.log_artifact(local_path=str(output_path.joinpath('dv.pkl')), artifact_path='sklearn-artifact')
