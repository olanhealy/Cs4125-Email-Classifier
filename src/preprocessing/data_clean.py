import pandas as pd
import re
from src.preprocessing.translate import translate_to_english

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        # Return an empty string if the input is not a string
        return ''  
    
    # Clean the text
    #TODO: Discuss with team if we need to remove more phrases
    text = re.sub(r'\bSent from my .*\b', '', text)
    text = re.sub(r'\bCustomer Support.*', '', text)
    return text

# Function to preprocess a DataFrame
def preprocess_data(df, output_filename):
    selected_columns = ['Ticket id', 'Interaction id', 'Ticket Summary', 'Interaction content', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
    # This copy was created to avoid SettingWithCopyWarning
    df_selected = df[selected_columns].copy() 

    # Translate and clean text data
    df_selected.loc[:, 'Interaction content'] = df_selected['Interaction content'].apply(translate_to_english)
    df_selected.loc[:, 'Interaction content'] = df_selected['Interaction content'].apply(clean_text)
    df_selected.fillna('Unknown', inplace=True)
    grouped_data = df_selected.groupby('Type 1').size().reset_index(name='Count')

    # Save the cleaned data to a new temp CSV file
    df_selected.to_csv(output_filename, index=False)
    print(f"Data cleaning and output completed for {output_filename}")

    return df_selected, grouped_data