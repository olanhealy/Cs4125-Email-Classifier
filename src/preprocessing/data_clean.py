import pandas as pd
import re

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ''  # Return an empty string if the input is not a string
    
    # Clean the text using regex substitutions
    # TODO: Discuss with the team what will need to be substituted
    text = re.sub(r'\bSent from my .*\b', '', text)
    text = re.sub(r'\bCustomer Support.*', '', text)
    return text

# Function to preprocess a DataFrame
def preprocess_data(df, output_filename):
    # Step 1: Select relevant columns
    # TODO: Discuss with the team what columns are relevant
    selected_columns = ['Ticket id', 'Interaction id', 'Ticket Summary', 'Interaction content', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
    df_selected = df[selected_columns]

    # Step 2: Group data by 'Type 1' to get counts
    grouped_data = df_selected.groupby('Type 1').size().reset_index(name='Count')

    # Step 3: Clean 'Interaction content'
    df_selected['Interaction content'] = df_selected['Interaction content'].apply(clean_text)

    # Step 4: Handle missing data by filling with placeholders
    df_selected.fillna('Unknown', inplace=True)

    # Step 5: Export the preprocessed data
    # TODO: Remove export and stream data to the next step
    df_selected.to_csv(output_filename, index=False)
    print(f"Data cleaning and output completed for {output_filename}")

    return df_selected, grouped_data

# Load and preprocess the AppGallery file
df_appgallery = pd.read_csv('data/AppGallery.csv')
cleaned_appgallery, grouped_appgallery = preprocess_data(df_appgallery, 'data/preprocessed_appgallery_data.csv')

# Load and preprocess the Purchasing file
df_purchasing = pd.read_csv('data/Purchasing.csv')
cleaned_purchasing, grouped_purchasing = preprocess_data(df_purchasing, 'data/preprocessed_purchasing_data.csv')
