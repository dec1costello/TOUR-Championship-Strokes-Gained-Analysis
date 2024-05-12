import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

def binnig(df: DataFrame, green_bool: bool, bin_by: str) -> DataFrame:
    """
    Perform binning of data based on a specified column.

    Parameters:
        df (DataFrame): The DataFrame containing the data.
        green_bool (bool): A boolean indicating whether to filter for 'Green' or not.
        bin_by (str): The column name to bin the data by.

    Returns:
        DataFrame: The DataFrame with an additional column containing bin labels.

    """

    if green_bool == False:
        temp_df = df[df['from_location_scorer'] != 'Green']
        Title = 'Non Green ' + bin_by
    else:
        temp_df = df[df['from_location_scorer'] == 'Green']
        Title = 'Green ' + bin_by
    plt.title(Title)
    B = plt.boxplot(temp_df[bin_by], vert=False)
    # plt.show()

    whisker_values = [whiskers.get_xdata() for whiskers in B['whiskers']]
    median_values = [medians.get_xdata() for medians in B['medians']]

    bin1 = (median_values[0][0] + whisker_values[0][0])/2
    bin2 = (median_values[0][0] + whisker_values[1][0])/2

    bins =    [0,       bin1,      bin2,601]
    labels = [f'0-{bin1}', f'{bin1}-{bin2}', f'{bin2}-601']

    if bins[0]==bins[1]:
        del labels[0]
        del bins[0]

    print(f'labels are {labels}')
    print(f'bins are {bins}')

    if green_bool == False:
        temp_df = df[df['from_location_scorer'] != 'Green']
        Title = 'non_putting_' + bin_by + '_bins'
    else:
        temp_df = df[df['from_location_scorer'] == 'Green']
        Title = 'putting_' + bin_by + '_bins'

    df[Title] = pd.cut(df[bin_by], bins=bins, labels=labels, right=False)
    df[Title] = df[Title].cat.add_categories([''])
    df[Title].fillna('', inplace=True)

    print(f'Added {Title} col')

    return df