import pandas as pd



def get_labels(df:pd.DataFrame,
               col_label:str,
               verbose:bool=True):
    """Extract unique labels from the dataframe and
    save them to a list. Print the number of labels in the dataset
    as well as the first 5 labels if there are more than five labels 
    in the dataset.

    Args:
        df (pd.DataFrame): Dataframe in which the labels are contained.
        col_label (str): Name of the column in the dataframe containing labels.
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



def count_tokens(df:pd.DataFrame, 
                 col_input_ids:str="input_ids", 
                 col_attn_mask:str=None):
    """Counts the number of tokens in each row of a DataFrame where the 
    attention mask is 1.

    Args:
        df (pd.DataFrame): Dataframe containing the token data.
        col_input_ids (str, optional): Name of the column in df that contains the input IDs. Defaults to "input_ids".
        col_attn_mask (str, optional): Name of the column in df that contains the attention masks. Defaults to None.
        
    Returns:
        pd.Series with the count of tokens for each row.
    
    Examples:
        df_train['cnt_tokens'] = count_tokens_in_dataframe(df_train)
    """
    
    def count_tokens(row):
        paired_tokens = zip(row['input_ids'], row['attention_mask'])
        return sum(1 for input_ids, mask in paired_tokens if mask == 1)
    
    return df.progress_apply(count_tokens, axis=1)