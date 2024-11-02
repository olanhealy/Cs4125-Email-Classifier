import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.data_clean import preprocess_data
from src.preprocessing.label import classify_with_reasoning

#clean and translate the data. This is then outputted to csv
df = pd.read_csv('data/AppGallery.csv')
df_cleaned, _ = preprocess_data(df, 'data/preprocessed_appgallery_data.csv')

#Apply labelling to the data 
df_cleaned['label'] = df_cleaned['Interaction content'].apply(classify_with_reasoning)

#Split the labeled data into training and testing sets, with the interaction content as the feature and the label as the target
X = df_cleaned['Interaction content']
y = df_cleaned['label']

#Split at 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The processed and split data is stored as a Dataframe
train_data = pd.DataFrame({'Interaction content': X_train, 'label': y_train})
test_data = pd.DataFrame({'Interaction content': X_test, 'label': y_test})

# Function to return the data. This will be called in the models
def get_processed_data():
    return train_data, test_data

print("Preprocessing: Data cleaning, translation, labeling, and splitting completed.")