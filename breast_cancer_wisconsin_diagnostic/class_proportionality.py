import pandas as pd
import matplotlib.pyplot as plt


def plot_class_distribution(df, dataset_name="Dataset"):

    class_labels = {'B': 'Benigno', 'M': 'Maligno'}
    class_counts = df['Diagnosis'].value_counts()
    rows = len(df)
    plt.figure(figsize=(8, 8))

    class_counts.plot.pie(
        labels=[class_labels.get(x, x) for x in class_counts.index],
        autopct=lambda p: f'{p:.2f}% ({p / 100 * rows:.0f})'
    )

    plt.title(f'Proporção das Classes - {dataset_name}')
    plt.ylabel('')
    plt.show()


train_filename = 'breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_train.csv'
test_filename = 'breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_test.csv'

train_df = pd.read_csv(train_filename)
test_df = pd.read_csv(test_filename)

plot_class_distribution(train_df, dataset_name="Treinamento")
plot_class_distribution(test_df, dataset_name="Teste")
