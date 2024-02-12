import pandas as pd
import datasets



def stratified_sample_from_dataset(data:datasets.DatasetDict, 
                                   random_seed:int, 
                                   perc_sample:float):
    """Sample indices using stratified sampling without replacement.
    In addition, save the indices of the complement of the set of 
    sampled indices.

    Args:
        dataset (DatasetDict): Dataset dictionary from which to sample.
        random_seed (int): Project arguments.
        perc_sample (float): percentage of samples to obtain

    Returns:
        list: list of indices
    """
    # Convert the dataset into a dataframe
    df = pd.DataFrame(data)
    # Sample without replacement
    train_idx = df.groupby('label', group_keys=False)\
             .apply(lambda x: x.sample(frac=perc_sample,
                                       random_state=random_seed,
                                       replace=False))\
             .index.values.tolist()
    # Get the complement indices
    cond = df.index.isin(train_idx)
    train_idxC = df[~cond].copy().index.values.tolist()
    
    return train_idx, train_idxC