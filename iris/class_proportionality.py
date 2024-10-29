import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_class_proportionality(filename):
    # Read the dataset from the given filename
    df = pd.read_csv(filename)

    # Calculate the proportionality of each class
    class_counts = df['class'].value_counts()

    rows = len(df)

    # Plotting the pie chart
    plt.figure(figsize=(8, 8))
    class_counts.plot.pie(
        startangle=90,
        colors=['#ff9999', '#66b3ff', '#99ff99'],
        autopct=lambda p: f'{p:.2f}% ({p / 100 * rows:.0f})'
    )
    plt.title('Class Proportionality')
    plt.ylabel('')  # Hide the y-label
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: class_proportionality.py <filename>")
    else:
        filename = sys.argv[1]
        plot_class_proportionality(filename)
