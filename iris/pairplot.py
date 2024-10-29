import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def plot_pairplot(filename):
    # Read the dataset from the given filename
    df = pd.read_csv(filename)

    # Create the pairplot
    sns.pairplot(df, hue='class')
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
    else:
        filename = sys.argv[1]
        plot_pairplot(filename)
