import os
import sys
from model import run_model

def list_files_and_models():
    # List available datasets
    datasets = [
        "AppGallery.csv",
        "preprocessed_appgallery_data.csv",
        "Purchasing.csv"
    ]
    print("Available CSV files:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}: {dataset}")
    
    dataset_choice = int(input("Select a CSV file by number: ")) - 1
    if dataset_choice < 0 or dataset_choice >= len(datasets):
        print("Invalid choice!")
        sys.exit(1)
    
    selected_dataset = datasets[dataset_choice]

    # List available models
    models = [
        "extra_trees",
        "gradient_boosting",
        "naive_bayes",
        "svm",
        "xg_boost"
    ]
    print("\nAvailable models:")
    for i, model in enumerate(models, 1):
        print(f"{i}: {model}")
    
    model_choice = int(input("Select a model by number: ")) - 1
    if model_choice < 0 or model_choice >= len(models):
        print("Invalid choice!")
        sys.exit(1)
    
    selected_model = models[model_choice]

    return selected_dataset, selected_model

if __name__ == "__main__":
    dataset, model_name = list_files_and_models()
    run_model(dataset, model_name)
