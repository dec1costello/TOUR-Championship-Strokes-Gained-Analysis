import pandas as pd

def tempGeoDf(df, hole):
    temp_df = df[df['hole'] == hole]
    temp_df = temp_df[temp_df['x'] != 0]
    temp_df = temp_df[temp_df['y'] != 0]
    temp_df = temp_df[temp_df['to_location_scorer'] != 'Unknown']

    return temp_df