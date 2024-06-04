from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export(output: tuple[LinearRegression, BaseEstimator]) -> None:
    model, _ = output
    print(model.intercept_)
