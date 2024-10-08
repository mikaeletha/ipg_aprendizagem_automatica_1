import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def create_boxplots(df):
    # Create a boxplot for each numeric column
    for column in df.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()


def main(file_path):
    # Read the DataFrame from the specified file
    df = pd.read_csv(file_path)

    # Create boxplots
    create_boxplots(df)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create boxplots to detect outliers in a dataset.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')

    # Parse arguments
    args = parser.parse_args()

    # Call main function with the specified file path
    main(args.file_path)
