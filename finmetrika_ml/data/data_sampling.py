import pandas as pd
import datasets



def stratified_sample_from_dataset(data:datasets.DatasetDict,
                                   by_split:str,
                                   random_seed:int, 
                                   perc_sample:float,
                                   return_complement_sample:bool=True):
    """Stratified sampling without replacement. Sample a percentage of a dataset given the dataset split.
    If 'return_complement_sample' is set to True then the function returns the complement sample as well.

    Args:
        dataset (DatasetDict): Dataset from which to sample. For example, my_dataset['train'].
        by_split (stt): Which data subset based on split should we sample from. Example: 'train'.
        random_seed (int): Project arguments.
        perc_sample (float): percentage of samples to obtain
        return_complement_sample (bool): Save the compleent sample as well.

    Returns:
        DatasetDict: selected sample, complement of selected sample (if return_complement_sample is True.)
    """
    data.set_format('pandas')
    # Convert the dataset into a dataframe
    df = data[by_split][:]
    
    # Sample without replacement
    train_idx = df.groupby('label', group_keys=False)\
             .apply(lambda x: x.sample(frac=perc_sample,
                                       random_state=random_seed,
                                       replace=False))\
             .index.values.tolist()
    
    # Select based on index
    data_idx = data[by_split].select(train_idx)
    
    
    if return_complement_sample:
        # Get the complement indices
        cond = df.index.isin(train_idx)
        train_idxC = df[~cond].copy().index.values.tolist()
        data_idxC = data[by_split].select(train_idxC)
        
        return datasets.DatasetDict({
            str(by_split): data_idx,
            str(by_split)+'C': data_idxC,
        })
        
    else:
        
        return datasets.DatasetDict({
            str(by_split): data_idx
        })