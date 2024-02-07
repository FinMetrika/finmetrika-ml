import pandas as pd



def get_labels(df:pd.DataFrame,
               col_label:str,
               verbose:bool=True):
    """Extract unique labels from the dataframe and
    save them to a list. Print the number of labels in the dataset
    as well as the first 5 labels if there are more than five labels 
    in the dataset.

    Args:
        df (pd.DataFrame): dataframe in which the labels are contained
        col_label (str): name of the column in the dataframe containing labels
        verbose (bool, optional): Print the statements. Defaults to True.
    """
    # Extract unique labels
    labels = df[col_label].unique().tolist()
    
    try:
        assert len(labels) == 1
    except AssertionError as e:
        print(f'Assertion error: {e}')
        
    if verbose:
        print(
            f'Number of labels: {len(labels)}',
            f'First 5 labels: {labels[:5]}' if len(labels) >= 5 else f'All labels: {labels}'
              )
        
    return labels