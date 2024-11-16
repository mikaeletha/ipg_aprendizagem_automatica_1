import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def remove_duplicate_rows(df):
    initial_count = len(df)
    df_cleaned = df.drop_duplicates()
    final_count = len(df_cleaned)

    duplicates_removed = initial_count - final_count

    if duplicates_removed > 0:
        print(f"{duplicates_removed} duplicate samples were removed.")
    else:
        print("No duplicate samples were found.")

    print(f"Dataset contains {final_count} Samples")

    return df_cleaned


column_names = ['ID', 'Diagnosis', 'Mean_Radius', 'Mean_Texture', 'Mean_Perimeter', 'Mean_Area', 'Mean_Smoothness', 'Mean_Compactness', 'Mean_Concavity', 'Mean_Concave_Points', 'Mean_Symmetry', 'Mean_Fractal_Dimension',
                'SE_Radius', 'SE_Texture', 'SE_Perimeter', 'SE_Area', 'SE_Smoothness', 'SE_Compactness', 'SE_Concavity', 'SE_Concave_Points', 'SE_Symmetry', 'SE_Fractal_Dimension',
                'Worst_Radius', 'Worst_Texture', 'Worst_Perimeter', 'Worst_Area', 'Worst_Smoothness', 'Worst_Compactness', 'Worst_Concavity', 'Worst_Concave_Points', 'Worst_Symmetry', 'Worst_Fractal_Dimension']

dataset = pandas.read_csv(
    "breast_cancer_wisconsin_diagnostic/data/wdbc.data", header=None, names=column_names)

dataset = remove_duplicate_rows(dataset)

print(dataset.describe(include="all"))

dataset.to_csv(
    "breast_cancer_wisconsin_diagnostic/pre_processed/wdbc.csv", index=False)

x = dataset.drop(columns=['Diagnosis'])
t = dataset['Diagnosis']

x_train, x_test, t_train, t_test = (
    train_test_split(x, t, train_size=0.5, stratify=t))

scaller = MinMaxScaler((-1, 1)).fit(x_train)

input_columns = column_names[:len(column_names) - 1]

x_train_scaled = pandas.DataFrame(
    scaller.transform(x_train), columns=x_train.columns, index=x_train.index
)
x_test_scaled = pandas.DataFrame(
    scaller.transform(x_test), columns=x_test.columns, index=x_test.index
)

train = pandas.concat([x_train_scaled, t_train], axis='columns', join='inner')
test = pandas.concat([x_test_scaled, t_test], axis='columns', join='inner')

train.to_csv(
    'breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_train.csv', index=False)
test.to_csv(
    'breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_test.csv', index=False)
