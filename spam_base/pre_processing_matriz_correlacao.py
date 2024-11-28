import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

column_names = column_names = [
    # Frequência de palavras
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    # Frequência de caracteres
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
    "char_freq_$", "char_freq_#",
    # Estatísticas de letras maiúsculas
    "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total",
    # Classe
    "spam"
]
dataset = pandas.read_csv(
    "spam_base/data/spambase.data", header=None, names=column_names)

# REMOVE LINHA DUPLICADAS
dataset = dataset.drop_duplicates()
print('REMOVE LINHA DUPLICADAS, OK')

# REMOVE LINHHAS VAZIAS
dataset = dataset.dropna()
print('REMOVE LINHHAS VAZIAS, OK')

# SALVA DADOS TRATADOS
dataset.to_csv(
    "spam_base/pre_processed/spambase.csv", index=False)
print('SALVA DADOS TRATADOS, OK')

# INFORMAÇÕES DO DATASET
print('INFORMAÇÕES DO DATASET')
print(dataset.describe(include="all"))

# SEPERAR FEATURES E TARGET
x = dataset.drop(columns=['spam'])  # FEATURES - O QUE VAI SER PREVISTO
t = dataset['spam']  # TARGET - CARACTERISTICAS

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
    'spam_base/pre_processed/spambase_train.csv', index=False)
test.to_csv(
    'spam_base/pre_processed/spambase_test.csv', index=False)

print("Processamento concluído! Os dados de treino e teste foram salvos com sucesso.")

# CALCULAR MATRIZ DE CORRELAÇÃO
# correlation_matrix = dataset.corr()
# print(correlation_matrix)
# correlation_matrix.to_csv(
#     'spam_base/pre_processed/correlation_matrix.csv')

# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True,
#             cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Matriz de Correlação das Variáveis')
# plt.show()
