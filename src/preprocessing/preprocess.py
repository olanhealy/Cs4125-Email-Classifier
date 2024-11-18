import pandas as pd
from preprocessing.preprocessor import preprocess_data_with_types

# Load raw data
df = pd.read_csv('data/AppGallery.csv')

# Preprocess and save to CSV
preprocessed_filename = 'data/preprocessed_appgallery_data_with_types.csv'
preprocess_data_with_types(df, preprocessed_filename)
