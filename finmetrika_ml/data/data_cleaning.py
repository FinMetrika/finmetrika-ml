import re
import pandas as pd


def remove_repeated_words(text:str):
    """Replace matched repeated words with a single instance of the word.

    Args:
        text (str): Input text.

    Returns:
        str: Input text with removed repeated words
    """
    # Pattern to match word boundaries followed by one or more non-whitespace
    # characters, followed by any number of whitespace, followed by the same word again
    pattern = r'\b(\w+)(\s+\1\b)+'
    
    text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
    
    return text


def remove_commas_and_extra_spaces(text:str):
    """Remove any comma within the text. 
       Remove any extra white spaces within and at the edges of text.

    Args:
        text (str): Text to apply the function.
    """
    # Remove commas
    text_no_commas = text.replace(",","")
    
    # Remove extra spaces
    text_single_space = re.sub(" +", " ", text_no_commas)

    return text_single_space.strip()


def remove_abrv(df:pd.DataFrame,
                text_column:str,
                abrv_patterns:list=None):
    """Remove abbreviations from text. Default abbreviations are: 
    [" D.O.O", "\.DE", "\.de", "\.COM", "S.R.L", "\.NET", "\.com", "\.co", "D.O", "\*"].
    To add any other ones to the list specify the 'abrv_patterns' argument.

    Args:
        text (str): Text to apply the function.
        
    """
    # Define initial abbreviations to remove
    patterns_list = [" D.O.O", "\.DE", "\.de", "\.COM", "S.R.L", "\.NET", "\.com", "\.co", "D.O", "\*"]
    
    # Add extra specified by the user
    if abrv_patterns is not None:
        patterns_list.extend(abrv_patterns)

    for p in patterns_list:
        df[text_column] = df[text_column].apply(lambda x: re.sub(p, '', x))

    return df

