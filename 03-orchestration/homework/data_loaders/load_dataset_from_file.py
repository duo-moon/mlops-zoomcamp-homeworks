import pathlib

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader


@data_loader
def load(**kwargs) -> pd.DataFrame:
    dataset_path = pathlib.Path(kwargs['configuration'].get('dataset_path'))
    return pd.read_parquet(dataset_path)
