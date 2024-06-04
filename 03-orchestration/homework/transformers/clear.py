import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data
        .assign(duration=lambda df: (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60)
        .loc[lambda df: (df['duration'] >= 1) & (df['duration'] <= 60)]
        .astype({'PULocationID': str, 'DOLocationID': str})
    )


@test
def test_output(output: pd.DataFrame) -> None:
    assert output.shape[0] == 3316216, f'Data should have 3316216 examples, but has {output.shape[0]}'
