import re

def clean_text(text):
    """
    Clean the input text by removing noise, special characters, and unnecessary spaces.

    :param text: The input text to clean.
    :return: The cleaned text.
    """
    if not isinstance(text, str):
        return ''

    # Remove dataset-specific noise
    text = re.sub(r'\bSent from my .*\b', '', text)  # Remove mobile signature
    text = re.sub(r'\bCustomer Support.*', '', text)  # Remove "Customer Support" mentions
    text = re.sub(r'\bRe:.*\b', '', text)  # Remove email reply prefixes
    text = re.sub(r'\b[^\w\s]\b', '', text)  # Remove special single characters
    text = text.strip()  # Remove trailing/leading spaces
    return text
