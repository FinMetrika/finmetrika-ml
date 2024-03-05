import pandas as pd
import torch
from finmetrika_ml.utils import *
from datasets import DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModel


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



def get_labels_from_dataset(dts:DatasetDict,
                       split:str,
                       label_column_name:str):
    """Get number of labels from the dataset.
    
    Args:
        dts (DatasetDict): Dataset with at least one split.
        split (str): Dataset sample, e.g. 'train'.
        label_column_name (str): Name of the column where labels are stored in the dts.
    """
    return dts[split].features[label_column_name].num_classes



def count_tokens(df:pd.DataFrame, 
                 col_input_ids:str="input_ids", 
                 col_attn_mask:str="attention_mask"):
    """Counts the number of tokens in each row of a DataFrame where the 
    attention mask is 1.

    Args:
        df (pd.DataFrame): Dataframe containing the token data.
        col_input_ids (str, optional): Name of the column in df that contains the input IDs. Defaults to "input_ids".
        col_attn_mask (str, optional): Name of the column in df that contains the attention masks. Defaults to None.
        
    Returns:
        pd.Series with the count of tokens for each row.
    
    Examples:
        df_train['cnt_tokens'] = count_tokens(df_train)
    """
    
    def count_tokens(row):
        paired_tokens = zip(row[col_input_ids], row[col_attn_mask])
        return sum(1 for _, mask in paired_tokens if mask == 1)
    
    return df.progress_apply(count_tokens, axis=1)



def tokenize(data_sample:DatasetDict,
             tokenizer:PreTrainedTokenizerBase,
             padding:str="max_length",
             text_column_name:str='text',
             truncation:bool=False,
             max_length:int=None):
    """Tokenize text in the column 'text_column_name'. Note that prior to applying the function
    the data_sample needs to be set to the 'torch' format by using data_sample.set_format('torch').
    TODO Update docstring
    Args:
        data_sample (DatasetDict): Dataset including input text.
        tokenizer (PreTrainedTokenizerBase): The tokenizer corresponding to the model, used to identify model input names.
        text_column_name (str): Name of the column which we wish to tokenize.

    Returns:
        input_ids, attention_mask: Tokenized text as input_ids and the corresponding attention masks if applicable.
    
    Examples:
        my_dataset.set_format('torch')
        my_dataset_enc = my_dataset.map(lambda batch: tokenize(batch, tokenizer, 'text'), 
                                            batched=True, batch_size=None)

    """
    return tokenizer(data_sample[text_column_name], 
                     padding=padding, 
                     truncation=truncation,
                     max_length = max_length,
                     add_special_tokens=True,
                     return_tensors="pt")
    


def extract_feature_vector(data_sample:DatasetDict, 
                           model:PreTrainedModel, 
                           tokenizer:PreTrainedTokenizerBase,
                           device:str):
    """Extract features from large language models for text classification.

    Args:
        data_sample (DatasetDict): Dataset including tokenized inputs. Expected to be a dictionary with keys matching the model's expected input names.
        model (PreTrainedModel): The model from which to extract the feature vectors. Should be an instance of a class derived from transformers.PreTrainedModel.
        tokenizer (PreTrainedTokenizerBase): The tokenizer corresponding to the model, used to identify model input names.
        device (str): Compute engine to which the inputs should be transfered. Define using check_device().

    Returns:
        - dict: A dictionary containing the feature vectors under the key "feature_vector".
    """
    
    inputs = {k:v.to(device) for k,v in data_sample.items() if k in tokenizer.model_input_names}
    with torch.inference_mode():
        # outputs.last_hidden_state.size() >>> [batch_size, n_tokens, hidden_dim]
        last_hidden_state = model(**inputs).last_hidden_state
    return {"feature_vector": last_hidden_state[:,0].cpu().numpy()}



class TRXDataset(torch.utils.data.Dataset):
    """Define the transaction dataset. Dataset should be of the form DatasetDict().
    Dataset should be tokenized and include at least "input_ids" and "text" column names.

    Args:
        dataset_split (DatasetDict): Dataset including input text and input_ids (tokenized text).
        device (str): Device on which to train the model. Use utils.check_device().
    """
    def __init__(self, dataset_split:DatasetDict,
                 device:str):
        self.dataset_split = dataset_split
        self.device = device
    
    def __getitem__(self, idx):
        return {k: torch.tensor(self.dataset_split[idx][k]).to(self.device)\
                    for k in self.dataset_split.features\
                    if k in ['input_ids', 'attention_mask', 'label']}
    
    def __len__(self):
        return len(self.dataset_split)
    


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