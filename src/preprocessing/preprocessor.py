import pandas as pd
from sklearn.model_selection import train_test_split
from .text_cleaner import clean_text
from .translator import translate_to_english, report_translation_rate

def preprocess_data_with_splits(df, save_output=False, output_filename=None, test_size=0.2, random_state=42):
    """
    Preprocess the dataset, clean and translate text, create hierarchical labels,
    and split the data into training and testing DataFrames.

    :param df: The input DataFrame containing the dataset.
    :param save_output: Flag to save the processed data to a CSV file.
    :param output_filename: Name of the output file if save_output is True.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Random seed for reproducibility.

    :return: Two DataFrames - train_data and test_data with cleaned and labeled data.
    """
    selected_columns = ['Ticket id', 'Interaction id', 'Ticket Summary', 
                        'Interaction content', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
    df_selected = df[selected_columns].copy()

    # Clean and translate text
    df_selected['Interaction content'] = df_selected['Interaction content'].apply(clean_text)
    df_selected['Interaction content'] = df_selected['Interaction content'].apply(translate_to_english)
    report_translation_rate()

    # Fill missing values
    df_selected.fillna('Unknown', inplace=True)

    # Create hierarchical label combining multiple type columns
    df_selected['hierarchical_label'] = df_selected[['Type 1', 'Type 2', 'Type 3', 'Type 4']].apply(
        lambda x: ' > '.join(x), axis=1
    )

    # Save preprocessed data to a file if required
    if save_output and output_filename:
        df_selected.to_csv(output_filename, index=False)
        print(f"Data cleaning and output completed for {output_filename}")

    # Split into training and testing sets
    X = df_selected['Interaction content']
    y = df_selected['hierarchical_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Combine X and y for train and test DataFrames
    train_data = pd.DataFrame({'Interaction content': X_train, 'label': y_train})
    test_data = pd.DataFrame({'Interaction content': X_test, 'label': y_test})

    return train_data, test_data
