import pandas as pd
import torch
from finmetrika_ml.utils import *


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



def extract_feature_vector(data_sample, model, tokenizer):
    
    # Get compute device
    device = check_device()
    
    def extract_feature_vector(batch):
        inputs = {k:v.to(device) for k,v in data_sample.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            # outputs.last_hidden_state.size() >>> [batch_size, n_tokens, hidden_dim]
            last_hidden_state = model(**inputs).last_hidden_state
        return {"feature_vector": last_hidden_state[:,0].cpu().numpy()}

    data_sample.set_format('torch')
    
    return data_sample.map(extract_feature_vector, batched=True)


class RegressionDataset1D(torch.utils.data.Dataset):
    
    def __init__(self, X, y):
        self.X = X.reshape(-1,1)
        self.y = y.reshape(-1,1)
    
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx,:],
                            dtype=torch.torch.float32),\
                torch.tensor(self.y[idx,:],
                             dype=torch.float32)
        
    
    def __len__(self):
        return self.X.shape[0]