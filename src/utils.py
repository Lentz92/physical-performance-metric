import pandas as pd
import polars as pl


def filter_by_frequency(df, frequency_threshold):
    # Count the occurrences of each player
    player_counts = df.group_by('player').len().select([
        pl.col('player'),
        pl.col('len').alias('player_count')
    ])

    # Filter players that appear less than the frequency threshold
    players_to_keep = player_counts.filter(pl.col('player_count') >= frequency_threshold).select('player')

    # Filter the original DataFrame to keep only the desired players
    df_filtered = df.filter(pl.col('player').is_in(players_to_keep['player']))

    return df_filtered


def split_time_series_data(df, split_criterion):
    """
    Splits the dataframe into training and test sets based on a date or a percentage.

    Parameters:
    df (pd.DataFrame): The dataframe to split. Must contain a 'date' column.
    split_criterion (float or str):
        - If a float, it represents the proportion of data to be used for the test set (e.g., 0.2 for 20%).
          The function will find the date that splits the dataframe into the desired proportion.
        - If a str, it represents the exact date for splitting the dataset (e.g., '2022-01-01').

    Returns:
    train_df (pd.DataFrame): The training set.
    test_df (pd.DataFrame): The test set.

    Raises:
    ValueError: If split_criterion is neither a float nor a string.
    """
    # Ensure the date column is of datetime type
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    if isinstance(split_criterion, float):
        # Sort the dataframe by date
        df = df.sort_values(by='date')

        # Determine the split index
        split_index = int(len(df) * (1 - split_criterion))

        # Split the data
        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]
    elif isinstance(split_criterion, str):
        # Split the data based on the date criterion
        train_df = df[df['date'] < split_criterion]
        test_df = df[df['date'] >= split_criterion]
    else:
        raise ValueError("split_criterion must be either a float (percentage) or a string (date).")

    return train_df, test_df
