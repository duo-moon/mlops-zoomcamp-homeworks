import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def train(data: pd.DataFrame) -> tuple[BaseEstimator, BaseEstimator]:
    features = ['PULocationID', 'DOLocationID']
    target = 'duration'

    dv = DictVectorizer()
    X_train = dv.fit_transform(data[features].to_dict(orient='records'))
    y_train = data[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    return lr, dv
