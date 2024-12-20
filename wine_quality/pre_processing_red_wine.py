import pandas as pd
from sklearn.model_selection import train_test_split


def remove_duplicate_rows(df):
    initial_count = len(df)
    df_cleaned = df.drop_duplicates()
    final_count = len(df_cleaned)
    duplicates_removed = initial_count - final_count

    if duplicates_removed > 0:
        print(f"{duplicates_removed} duplicate samples were removed.")
    else:
        print("No duplicate samples were found.")

    print(f"Dataset contains {final_count} samples")

    return df_cleaned


dataset = pd.read_csv("wine_quality/data/winequality-red.csv", delimiter=';')

dataset = remove_duplicate_rows(dataset)

# Verificar se há valores NaN e removê-los
if dataset.isnull().values.any():
    print("Missing values found. Removing missing values...")
    dataset = dataset.dropna()

print(dataset.describe(include="all"))

dataset.to_csv("wine_quality/pre_processed/wine_quality_red.csv", index=False)

x = dataset.drop(columns=['quality'])
t = dataset['quality']

x_train, x_test, t_train, t_test = train_test_split(
    x, t, train_size=0.5, stratify=t)

train = pd.concat([x_train, t_train], axis='columns', join='inner')
test = pd.concat([x_test, t_test], axis='columns', join='inner')

train.to_csv(
    "wine_quality/pre_processed/wine_quality_red_train.csv", index=False)
test.to_csv("wine_quality/pre_processed/wine_quality_red_test.csv", index=False)
