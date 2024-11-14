import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_boxplots(df, columns):
    for column in columns:
        # def create_boxplots(df):
        #     for column in df.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()


# def main(file_path):
#     df = pd.read_csv(file_path)
#     create_boxplots(df)

def main(file_path, columns):
    df = pd.read_csv(file_path)
    create_boxplots(df, columns)


file_path = 'breast_cancer_wisconsin_diagnostic/pre_processed/wdbc.csv'
columns = ['Diagnosis', 'radius1', 'texture1']

main(file_path, columns)
