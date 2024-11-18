import pandas as pd
from sklearn.model_selection import train_test_split
from .text_cleaner import clean_text
from .translator import translate_to_english, report_translation_rate

def preprocess_data_with_splits(df, output_filename, test_size=0.2, random_state=42):
    """
    Preprocess the dataset, clean and translate text, create hierarchical labels,
    and split the data into training and testing sets.
    """
    selected_columns = ['Ticket id', 'Interaction id', 'Ticket Summary', 
                        'Interaction content', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
    df_selected = df[selected_columns].copy()

    df_selected['Interaction content'] = df_selected['Interaction content'].apply(clean_text)
    df_selected['Interaction content'] = df_selected['Interaction content'].apply(translate_to_english)
    report_translation_rate()

    # Fill missing values
    df_selected.fillna('Unknown', inplace=True)

    # Create hierarchical label
    df_selected['hierarchical_label'] = df_selected[['Type 1', 'Type 2', 'Type 3', 'Type 4']].apply(
        lambda x: ' > '.join(x), axis=1
    )

    df_selected.to_csv(output_filename, index=False)
    print(f"Data cleaning and output completed for {output_filename}")

    X = df_selected['Interaction content']
    y = df_selected['hierarchical_label']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test