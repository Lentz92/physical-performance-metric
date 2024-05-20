from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import polars as pl
import numpy as np


def compute_performance_score(df, weights):
    """
    Compute the performance score based on the given weights.

    Parameters:
    df (polars.DataFrame): The input dataframe.
    weights (dict): Dictionary of feature weights.

    Returns:
    df (polars.DataFrame): The dataframe with the computed performance score.
    """
    df = df.with_columns([
        sum([pl.col(f'{col}_per_min') * weight for col, weight in weights.items() if col != 'maximum_velocity_km_h'] +
            [pl.col('maximum_velocity_km_h') * weights['maximum_velocity_km_h']]).alias('performance_score')
    ])
    return df


class NormalizePerformanceScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that normalizes performance scores within each position group using MinMax scaling.

    This transformer scales the `performance_score` column values for each unique `position`
    to a range between 0 and 100. The scaled scores are then rounded to the nearest integer.

    Attributes:
        scalers (dict): Dictionary where keys are position labels and values are fitted MinMaxScaler instances.

    Methods:
        fit(X, y=None): Fits a MinMaxScaler for each unique position in the input data.
        transform(X): Transforms the `performance_score` for each position using the fitted scalers and rounds the results.
    """

    def __init__(self):
        # Initialize a dictionary to store a scaler for each position
        self.scalers = {}

    def fit(self, X, y=None):
        # Create a separate scaler for each position and fit it
        positions = X['position'].unique()
        for position in positions:
            scaler = MinMaxScaler(feature_range=(0, 100))
            scaler.fit(X[X['position'] == position][['performance_score']])
            self.scalers[position] = scaler
        return self

    def transform(self, X):
        # Apply the appropriate scaler and round the results
        X = X.copy()
        for position, scaler in self.scalers.items():
            mask = X['position'] == position
            X.loc[mask, 'performance_score'] = scaler.transform(X[mask][['performance_score']])
            # Round the transformed scores to the nearest integer
            X.loc[mask, 'performance_score'] = np.round(X.loc[mask, 'performance_score']).astype(int)
        return X


class LaggedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that generates lagged features and rolling means for time series data.

    This transformer creates lagged features and rolling means for specified columns in a time series dataset.
    The features are calculated for each player individually.

    Attributes:
        weights (dict): Dictionary specifying the columns to transform.
        window_size (int): The window size for calculating rolling means, default is 7.

    Methods:
        fit(X, y=None): This transformer does not require fitting and returns itself.
        transform(X): Generates lagged features and rolling means for the specified columns.
    """

    def __init__(self, weights, window_size=7):
        self.weights = weights
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.sort_values(by=["player", "date"], ascending=True)
        for col in self.weights.keys():
            if col != 'maximum_velocity_km_h':
                X[f'{col}_per_min_rolling_mean_{self.window_size}'] = X.groupby('player')[f'{col}_per_min'].rolling(
                    self.window_size).mean().reset_index(level=0, drop=True)
                for i in range(1, 8):
                    X[f'{col}_per_min_lag_{i}'] = X.groupby('player')[f'{col}_per_min'].shift(i)
            else:
                X[f'maximum_velocity_km_h_rolling_mean_{self.window_size}'] = X.groupby('player')[
                    'maximum_velocity_km_h'].rolling(self.window_size).mean().reset_index(level=0, drop=True)
                for i in range(1, 8):
                    X[f'maximum_velocity_km_h_lag_{i}'] = X.groupby('player')['maximum_velocity_km_h'].shift(i)
        X = X.dropna().reset_index(drop=True)
        return X


class ShiftPerformanceScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that shifts the performance score to create a target variable for the next time step.

    This transformer sorts the data by player and date, and then shifts the `performance_score` column
    to create a new column `performance_score_next` which represents the performance score of the next time step.

    Methods:
        fit(X, y=None): This transformer does not require fitting and returns itself.
        transform(X): Shifts the `performance_score` to create the `performance_score_next` column.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.sort_values(by=["player", "date"], ascending=True).copy()
        X['performance_score_next'] = X.groupby('player')['performance_score'].shift(-1)
        return X.dropna().reset_index(drop=True)
