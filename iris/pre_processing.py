import pandas
import matplotlib.pyplot as plt
import seaborn

def remove_duplicate_rows(df):
    # Count the number of rows before removing duplicates
    initial_count = len(df)

    # Remove duplicates
    df_cleaned = df.drop_duplicates()

    # Count the number of rows after removing duplicates
    final_count = len(df_cleaned)

    # Calculate the number of duplicates removed
    duplicates_removed = initial_count - final_count

    # Print the number of duplicates removed
    if duplicates_removed > 0:
        print(f"{duplicates_removed} duplicate samples were removed.")
    else:
        print("No duplicate samples were found.")

    print(f"Dataset contains {final_count} Samples")

    return df_cleaned

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

dataset = pandas.read_csv("data/iris.data", header=None, names = column_names)

dataset = remove_duplicate_rows(dataset)

print(dataset.describe(include="all"))

num_variables = len(dataset.columns)
num_output_variables = 1
number_input_variables = num_variables - num_output_variables

input_columns = dataset.columns[0:number_input_variables]

#seaborn.pairplot(dataset, hue='class')
#plt.show()

dataset.to_csv("pre_processed/iris.csv", index=False)