import pandas as pd
from pandas import DataFrame

def tempGeoDf(df: DataFrame, hole: int) -> DataFrame:
    """
    Filter a DataFrame to extract data related to a specific hole, removing zero coordinates and unknown locations.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        hole (int): The hole number to filter the data for.

    Returns:
        DataFrame: The filtered DataFrame containing data related to the specified hole.
    """
    
    temp_df = df[df['hole'] == hole]
    temp_df = temp_df[temp_df['x'] != 0]
    temp_df = temp_df[temp_df['y'] != 0]
    temp_df = temp_df[temp_df['to_location_scorer'] != 'Unknown']

    return temp_df