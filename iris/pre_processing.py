import pandas
from sklearn.model_selection import train_test_split


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

dataset.to_csv("pre_processed/iris.csv", index=False)

x = dataset.drop(columns=['class'])
t = dataset['class']

x_train, x_test, t_train, t_test = (
    train_test_split(x, t, train_size=0.5, stratify=t))

train = pandas.concat([x_train, t_train], axis='columns', join='inner')
test = pandas.concat([x_test, t_test], axis='columns', join='inner')

train.to_csv('pre_processed/iris_train.csv', index=False)
test.to_csv('pre_processed/iris_test.csv', index=False)