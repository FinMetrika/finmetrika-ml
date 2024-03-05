import pandas as pd

def create_datetime_features(df:pd.DataFrame,
                              datetime_column:str):
    """Create features from datetime object. Current supported features are:
    - month,
    - year,
    - week day,
    - week in a year

    Args:
        df (pd.DataFrame): Dataframe containing the transaction dates.
        datetime_column (str): Column in the dataframe containing date.
    """
    
    # date in format Name of the Month Day Year
    df['DT_DATE_STR'] = df[datetime_column].apply(lambda x: x.strftime('%B %d %Y'))
    
    # Month of the year >>> 1, 2, ...
    df['DT_MONTH'] = df[datetime_column].dt.month
    # df['DT_MONTH'] = df[datetime_column].apply(lambda x: x.strftime('%m')).astype(int)
    
    # Month of the year in text >>> January, February, etc
    df['DT_MONTH_TXT'] = df[datetime_column].apply(lambda x: x.strftime('%B'))
    
    df['DT_YEAR'] = df[datetime_column].dt.year
    df['DT_WEEK_DAY'] = df[datetime_column].dt.weekday + 1 # Monday=1, Sunday=7
    df['DT_DAY_OF_YEAR'] = df[datetime_column].dt.day_of_year
    df['DT_WEEK_OF_YEAR'] = df[datetime_column].dt.isocalendar().week
    
    return df



def quantize_amount(df:pd.DataFrame,
                    txt_amount_column:str):
    """_summary_

    Args:
        df (pd.DataFrame): Dataframe containing the transaction amounts.
        txt_amount_column (str): Column in the dataframe containing the transaction amounts.
    """
    
    df['TRX_AMOUNT_BIN'] = pd.cut(df['TRX_AMOUNT'], 
                                        bins=[0, 50, 500, 1000, float('inf')],
                                        labels=['low', 'medium', 'high', 'luxury'],
                                        right=False)