import re
import pandas as pd
import numpy as np



# Define credit card pattern: 462765XXXXXX1234
#credit_card_pattern = r'(\d{6}X+\d{4})'

patterns_dict = {
    # pattern: 462765XXXXXX1234
    "credit_card_no" : r'(\d{6}X+\d{4})',
    
    # pattern for matching repeated words (case-insensitive)
    #"repeated_words" : r'\b(\w+)(\s+\1\b)+',
    
    "abbreviations"  : [r" d.d.", " DD", " D.O.O.", " d.o.o.", "DOO", "doo", r"\.DE", "\\.de", "\\.COM", "\\.com",
                        "S.R.L", "\\.NET", "\\.co", "D.O", "\\*", "WWW", r'\.EU'],
    
    "punctuation"    : [r"\.", r"\:", r"\'",
                        r"^\s*,\s",   # Remove the first comma (and any leading whitespace)
                        r",\s*$",   # Remove the last comma (and any trailing whitespace)
                        ],
    
    "non_ascii_chr"  : r'[^ -~]',
    
    # Match 'PBZ' at the start of the string
    "cro_abrv"       : [r"PBZT",   # 'PBZT'
                        r"PBZ\d",  # 'PBZ followed by a digit
                        r"TN\d+",   # 'TN' followed by any number of digits (Konzum)
                        r"T\d+\s",  # 'T' followed by any number of digits
                        r"P-\d+\s", # 'P-' followed by any number of digits and space (Tommy)
                        ],
    
    # Match P-0980, P-1234 for different branch of the store
    "branch-no"      : r"P-\d{4}",
    #TODO Add address from this number (see http://www.panteongroup.org/docs/ean-kode_Konzum.txt)
    
    # Remove numbers after ATM in 'ATM A3122001' >>> 'ATM'
    # pattern matches 'ATM' followed by an optional letter and a sequence of digits
    'atm_no'         : r"(ATM\s)[A-Za-z]?\d+\s",
    
    #TODO Extract and remove address and city form transaction
    #TODO IBAN info: "Prijenos sa HR1234 TEA "
    
    # Remove sa HR1234... 
    'iban'           : r"sa HR\d+\s"
    #TODO Should we also remove 'Prijenos ' from text
    #TODO What about 'Prijenos sa 12345678 Naplata s avista racuna NAME'
}


def remove_ccard(text:str) -> str:
    """Remove he masked credit card number from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Original text without the masked credit card.
    """
    
    pattern = patterns_dict['credit_card_no']
    
    return re.sub(pattern, '', text).strip()
    


def remove_repeated_words(text:str):
    """Iteratively check for repeated words in the text. Remove all repeated word
    instances.

    Args:
        text (str): Input text.

    Returns:
        str: Input text with removed repeated words.
    """
    
    seen = set()    # Keep track of already seen words
    result = []     # Keep the final sequence of non-repeating words
    
    # Split text into words
    for word in text.split():
        # Check is already seen (use lowercase to not have an issue with casing)
        if word.lower() not in seen:
            seen.add(word.lower())  # Add the lowercase version of the word to seen
            result.append(word)     # Add the original word
    
    return ' '.join(result)
    #return re.sub(pattern, r'\1', text, flags=re.IGNORECASE)



def remove_abrv_chr(text:str
                    #, abrv_patterns:list=None
                    ):
    """Remove abbreviations from text. To add any other ones to the list specify the 'abrv_patterns' argument.

    Args:
        text (str): Input text.
        abrv_patterns (list): Additional list of abberviations to be removed from text. 
    """
    
    # Define initial abbreviations to remove and make a copy
    patterns_list = patterns_dict["abbreviations"]#[:]
    
    # Add extra specified by the user
    # if abrv_patterns is not None:
    #     patterns_list.extend(abrv_patterns)

    modified_text = text
    for p in patterns_list:
        modified_text = re.sub(p, '', modified_text) #if isinstance(modified_text, str) else modified_text
        
    return modified_text.strip()



def remove_non_ascii_chr(text:str) -> str:
    """Replace non-ASCII characters with an empty string in the 'Text' column.

    Args:
        text (str): Input text.
    
    Returns:
        str: Input text with removed non-ASCII characters.
    """

    # Regex pattern to match non-ASCII characters
    pattern = patterns_dict["non_ascii_chr"]
    
    return re.sub(pattern, '', text)


def remove_cro_abrv(text:str) -> str:
    """Remove Croatian specific abbreviations: PBZT

    Args:
        text (str): Input text.
    
    Returns:
        str: Input text with removed "PBZT"
    """
    
    pattern = patterns_dict['cro_abrv']
    
    modified_text = text
    for p in pattern:
        modified_text = re.sub(p, '', modified_text) if isinstance(modified_text, str) else modified_text
    
    return modified_text



def remove_branch_info(text:str) -> str:
    """Remove branch info number like P-1234 from text.

    Args:
        text (str): Input text.
    
    Returns:
        str: Input text with removed branch info number.
    """
    pattern = patterns_dict["branch-no"]
    
    return re.sub(pattern, '', text)



def remove_atm_no(text:str) -> str:
    """Remove ATM numbers from text.

    Args:
        text (str): Input text.
    
    Returns:
        str: Input text with removed ATM numbers.
    """
    pattern = patterns_dict["atm_no"]
    # \1 refers to the first captured group (ATM and any spaces after it)
    return re.sub(pattern, r'\1', text)


def remove_iban(text:str) -> str:
    """Remove "sa HR1234..." from text.

    Args:
        text (str): Input text.
    
    Returns:
        str: Input text with removed IBAN numbers.
    """
    
    pattern = patterns_dict["iban"]
    
    return re.sub(pattern, '', text)



def remove_punctuation(text:str) -> str:
    """Remove any strain punctuation.

    Args:
        text (str): Input text.

    Returns:
        str: Input text without specified punctuations.
    """
    pattern = patterns_dict["punctuation"]
    
    modified_text = text
    for p in pattern:
        modified_text = re.sub(p, '', modified_text)
    
    return modified_text

