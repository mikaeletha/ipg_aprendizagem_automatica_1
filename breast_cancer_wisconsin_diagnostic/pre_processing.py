import pandas as pd
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split


def remove_duplicate_rows(df):
    initial_count = len(df)
    df_cleaned = df.drop_duplicates()
    final_count = len(df_cleaned)
    duplicates_removed = initial_count - final_count

    if duplicates_removed > 0:
        print(f"{duplicates_removed} duplicatas removidas.")
    else:
        print("Nenhuma amostra duplicada encontrada.")

    return df_cleaned


def remove_missing_values(dataset):
    if dataset.isnull().values.any():
        print("Valores ausentes encontrados. Removendo valores ausentes...")
        dataset = dataset.dropna()
    else:
        print("Nenhum valor ausente encontrado.")
    return dataset


column_names = ['ID', 'Diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2',
                'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']

dataset = pd.read_csv(
    "breast_cancer_wisconsin_diagnostic/data/wdbc.data", header=None, names=column_names)

dataset = remove_duplicate_rows(dataset)
dataset = remove_missing_values(dataset)

print(dataset.describe(include="all"))

x = dataset.drop(columns=['Diagnosis'])
t = dataset['Diagnosis']

x_train, x_test, t_train, t_test = train_test_split(
    x, t, train_size=0.5, stratify=t)

train = pd.concat([x_train, t_train], axis='columns', join='inner')
test = pd.concat([x_test, t_test], axis='columns', join='inner')

train.to_csv(
    "breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_train.csv", index=False)
test.to_csv(
    "breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_test.csv", index=False)
dataset.to_csv(
    "breast_cancer_wisconsin_diagnostic/pre_processed/wdbc.csv", index=False)
