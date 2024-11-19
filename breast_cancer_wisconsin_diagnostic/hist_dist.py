import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def create_displots(df):
    # Create a distribution plot for each numeric column
    for column in df.select_dtypes(include=['number']).columns:
        sns.displot(df, x=column, kde=True, hue='class', multiple='stack')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


def main(file_path):
    # Read the DataFrame from the specified file
    df = pd.read_csv(file_path)

    # Create distribution plots
    create_displots(df)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create distribution plots from a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')

    # Parse arguments
    args = parser.parse_args()

    # Call main function with the specified file path
    main(args.file_path)
