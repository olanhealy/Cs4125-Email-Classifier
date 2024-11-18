import pandas as pd
from src.preprocessing.preprocessor import preprocess_data_with_types

df = pd.read_csv('data/AppGallery.csv')

preprocessed_filename = 'data/preprocessed_appgallery_data_with_types.csv'
preprocess_data_with_types(df, preprocessed_filename)
