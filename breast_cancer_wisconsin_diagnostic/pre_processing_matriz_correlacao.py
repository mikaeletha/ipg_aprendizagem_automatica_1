import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

column_names = ['ID', 'Diagnosis', 'Mean_Radius', 'Mean_Texture', 'Mean_Perimeter', 'Mean_Area', 'Mean_Smoothness', 'Mean_Compactness', 'Mean_Concavity', 'Mean_Concave_Points', 'Mean_Symmetry', 'Mean_Fractal_Dimension',
                'SE_Radius', 'SE_Texture', 'SE_Perimeter', 'SE_Area', 'SE_Smoothness', 'SE_Compactness', 'SE_Concavity', 'SE_Concave_Points', 'SE_Symmetry', 'SE_Fractal_Dimension',
                'Worst_Radius', 'Worst_Texture', 'Worst_Perimeter', 'Worst_Area', 'Worst_Smoothness', 'Worst_Compactness', 'Worst_Concavity', 'Worst_Concave_Points', 'Worst_Symmetry', 'Worst_Fractal_Dimension']
dataset = pandas.read_csv(
    "breast_cancer_wisconsin_diagnostic/data/wdbc.data", header=None, names=column_names)

# MAPEIA A VARIAVEL DIAGNOSIS PARA VALORES BINARIOS
dataset['Diagnosis'] = dataset['Diagnosis'].map({'M': 1, 'B': 0})

# REMOVE LINHA DUPLICADAS
dataset = dataset.drop_duplicates()
print('REMOVE LINHA DUPLICADAS, OK')

# REMOVE LINHHAS VAZIAS
dataset = dataset.dropna()
print('REMOVE LINHHAS VAZIAS, OK')

# REMOVE COLUNAS IRRELEVANTES
dataset = dataset.drop(columns=['ID'])
dataset = dataset.loc[:, ~dataset.columns.str.startswith('SE_')]
print('REMOVE COLUNAS IRRELEVANTES, OK')

# SALVA DADOS TRATADOS
dataset.to_csv(
    "breast_cancer_wisconsin_diagnostic/pre_processed/wdbc.csv", index=False)
print('SALVA DADOS TRATADOS, OK')

# INFORMAÇÕES DO DATASET
print('INFORMAÇÕES DO DATASET')
print(dataset.describe(include="all"))

# SEPERAR FEATURES E TARGET
x = dataset.drop(columns=['Diagnosis'])  # FEATURES - O QUE VAI SER PREVISTO
t = dataset['Diagnosis']  # TARGET - CARACTERISTICAS

# SEPARAR EM TREINAMENTO E TESTE
x_train, x_test, t_train, t_test = (
    train_test_split(x, t, train_size=0.5, stratify=t))

# COLOCAR OS DADOS NUMÉRICOS NA MESMA ESCALA
scaller = MinMaxScaler((-1, 1)).fit(x_train)

x_train_scaled = scaller.transform(x_train)
x_test_scaled = scaller.transform(x_test)

# CRIAR DATAFRAME
x_train_scaled = pandas.DataFrame(
    x_train_scaled, columns=x_train.columns, index=x_train.index)
x_test_scaled = pandas.DataFrame(
    x_test_scaled, columns=x_test.columns, index=x_test.index)

train = pandas.concat([x_train_scaled, t_train], axis='columns', join='inner')
test = pandas.concat([x_test_scaled, t_test], axis='columns', join='inner')

# SALVAR EM CSV OS DADOS DE TREINO E TESTE
train.to_csv(
    'breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_train.csv', index=False)
test.to_csv(
    'breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_test.csv', index=False)

print("Processamento concluído! Os dados de treino e teste foram salvos com sucesso.")

# CALCULAR MATRIZ DE CORRELAÇÃO
correlation_matrix = dataset.corr()
print(correlation_matrix)
correlation_matrix.to_csv(
    'breast_cancer_wisconsin_diagnostic/pre_processed/correlation_matrix.csv')

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação das Variáveis')
plt.show()
