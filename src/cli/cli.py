import os
import sys
from src.cli.helper import helper

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



def list_files_and_models():
    # List available datasets
    datasets = [
        "AppGallery.csv",
        "preprocessed_appgallery_data.csv",
        "Purchasing.csv"
    ]
    print("\nAvailable CSV files:")
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

    formats = ["text", "json", "table"]
    print("\nAvailable result formats:")
    for i, fmt in enumerate(formats, 1):
        print(f"{i}: {fmt}")

    format_choice = int(input("Select a result format by number: ")) - 1
    if format_choice < 0 or format_choice >= len(formats):
        print("Invalid choice!")
        sys.exit(1)

    selected_format = formats[format_choice]

    return selected_dataset, selected_model, selected_format

def main():
    while True:
        print("\n\033[1m Email Classifier CLI \033[0m")
        print("1. Run classification")
        print("2. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            dataset, model_name, result_format = list_files_and_models()

            # Check if valid selections were made
            if dataset and model_name and result_format:
                print(f"\nRunning with CSV: {dataset}, Model: {model_name}, Format: {result_format}")
                helper(dataset, model_name, result_format)
            else:
                print("Invalid input. Returning to menu.")

        elif choice == "2":
            print("Exiting CLI. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
   main()

