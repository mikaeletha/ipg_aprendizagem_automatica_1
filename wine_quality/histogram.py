import pandas as pd
import matplotlib.pyplot as plt
import argparse


def create_histograms(df):
    # Create histograms for each column
    df.hist(bins=10, edgecolor='black', grid=False, figsize=(10, 5))

    # Add titles and labels
    plt.suptitle('Histograms for Each Variable')
    plt.show()


def main(file_path):
    # Read the DataFrame from the specified file
    df = pd.read_csv(file_path)

    # Create histograms
    create_histograms(df)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create histograms from a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')

    # Parse arguments
    args = parser.parse_args()

    # Call main function with the specified file path
    main(args.file_path)
