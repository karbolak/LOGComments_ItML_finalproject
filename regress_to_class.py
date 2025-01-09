import pandas as pd

def process_csv_column(file_path, column_name, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Ensure the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the CSV file.")

    # Apply the transformation: 1 if > 0.5, 0 otherwise
    df[column_name] = df[column_name].apply(lambda x: 1 if x > 0.25 else 0)

    # Save the modified DataFrame back to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Processed CSV saved to {output_file}")

# Example usage
input_csv = "datasets/activity_botscore_train.csv"       # Path to the input CSV file
output_csv = "datasets/activity_botscore_train_fixed.csv"     # Path for the output CSV file
column_to_process = "bot_score_english"  # Replace with your actual column name

process_csv_column(input_csv, column_to_process, output_csv)