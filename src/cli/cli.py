import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cli.helper import helper

def list_files_and_models():
    # List available datasets
    datasets = [
        "AppGallery.csv",
        "Purchasing.csv"
    ]

    while True:
        print("\nAvailable CSV files:")
        for i, dataset in enumerate(datasets, 1):
            print(f"{i}: {dataset}")
        
        try:
            dataset_choice = int(input("Select a CSV file by number (e.g., 1 or 2): ")) - 1
            if 0 <= dataset_choice < len(datasets):
                selected_dataset = datasets[dataset_choice]
                break
            else:
                print("Invalid choice! Please enter a number corresponding to the datasets listed (e.g., 1 or 2).")
        except ValueError:
            print("Invalid input! Please enter a valid number.")

    # List available models
    models = [
        "extra_trees",
        "gradient_boosting",
        "naive_bayes",
        "svm",
        "logistic_regression"
    ]

    while True:
        print("\nAvailable models:")
        for i, model in enumerate(models, 1):
            print(f"{i}: {model}")
        
        try:
            model_choice = int(input("Select a model by number (e.g., 1 or 2): ")) - 1
            if 0 <= model_choice < len(models):
                selected_model = models[model_choice]
                break
            else:
                print("Invalid choice! Please enter a number corresponding to the models listed (e.g., 1 or 2).")
        except ValueError:
            print("Invalid input! Please enter a valid number.")

    formats = ["text", "json", "table"]

    while True:
        print("\nAvailable result formats:")
        for i, fmt in enumerate(formats, 1):
            print(f"{i}: {fmt}")
        
        try:
            format_choice = int(input("Select a result format by number (e.g., 1 or 2): ")) - 1
            if 0 <= format_choice < len(formats):
                selected_format = formats[format_choice]
                break
            else:
                print("Invalid choice! Please enter a number corresponding to the result formats listed (e.g., 1 or 2).")
        except ValueError:
            print("Invalid input! Please enter a valid number.")

    return selected_dataset, selected_model, selected_format


def main():
    while True:
        print("\n\033[1m Email Classifier CLI \033[0m")
        print("1. Run classification")
        print("2. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            dataset, model_name, result_format = list_files_and_models()

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
