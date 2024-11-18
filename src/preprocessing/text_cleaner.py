import re

def clean_text(text):
    if not isinstance(text, str):
        return ''

    # Remove dataset-specific noise
    text = re.sub(r'\bSent from my .*\b', '', text)
    text = re.sub(r'\bCustomer Support.*', '', text)
    text = re.sub(r'\bRe:.*\b', '', text)  # Remove email reply prefixes
    text = re.sub(r'\b[^\w\s]\b', '', text)  # Remove special single characters
    text = text.strip()  # Remove trailing/leading spaces
    return text
