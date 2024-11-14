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


# "Mean" (média), "SE" (erro padrão) e "Worst" (maior valor)
# a média, o erro padrão e o "pior" valor (maior valor entre os três maiores)
column_names = [
    'ID', 'Diagnosis', 'Mean_Radius', 'Mean_Texture', 'Mean_Perimeter', 'Mean_Area', 'Mean_Smoothness', 'Mean_Compactness', 'Mean_Concavity', 'Mean_Concave_Points', 'Mean_Symmetry', 'Mean_Fractal_Dimension', 'SE_Radius', 'SE_Texture', 'SE_Perimeter', 'SE_Area', 'SE_Smoothness', 'SE_Compactness', 'SE_Concavity', 'SE_Concave_Points', 'SE_Symmetry', 'SE_Fractal_Dimension', 'Worst_Radius', 'Worst_Texture', 'Worst_Perimeter', 'Worst_Area', 'Worst_Smoothness', 'Worst_Compactness', 'Worst_Concavity', 'Worst_Concave_Points', 'Worst_Symmetry', 'Worst_Fractal_Dimension'
]

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
