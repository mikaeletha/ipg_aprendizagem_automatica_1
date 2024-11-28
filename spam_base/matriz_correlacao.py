import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('spam_base/pre_processed/spambase.csv')

# Calcular a matriz de correlação
correlation_matrix = dataset.corr()

# Exibir a matriz de correlação no console
print(correlation_matrix)

# Salvar a matriz de correlação como um arquivo CSV
correlation_matrix.to_csv('spam_base/pre_processed/correlation_matrix.csv')

# Plotar a matriz de correlação
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title('Matriz de Correlação das Variáveis')
plt.show()
