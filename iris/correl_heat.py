import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def plot_correlation_heatmap(filename):
    # Read the dataset from the given filename
    df = pd.read_csv(filename)

    # Convert text columns to numerical using one-hot encoding
    df_encoded = pd.get_dummies(df)

    # Calculate the correlation matrix
    corr_matrix = df_encoded.corr() * 100

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: correl_heat.py <filename>")
    else:
        filename = sys.argv[1]
        plot_correlation_heatmap(filename)
