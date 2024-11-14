import matplotlib.pyplot as plt
import pandas as pd
from numpy import sort
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier


# Função para exibir a matriz de confusão
def display_confusion_matrix(targets, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(
        targets, predicted, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()


# Função para carregar o dataset
def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    x = dataset.drop(columns=['Diagnosis'])
    t = dataset['Diagnosis']
    return x, t


# Função para treinar o modelo KNN
def train_knn(x, t, n_neighbors, p):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)
    knn.fit(x, t)
    return knn


# Função para avaliar o modelo
def evaluate_model(knn, x, t, data_type):
    y = knn.predict(x)
    accuracy = accuracy_score(t, y)
    print(f"\n{data_type} Accuracy: {accuracy * 100:.2f}%")

    # Relatório de classificação
    print(f"\n{data_type} Classification Report:")
    print(classification_report(t, y, digits=4))

    # Exibindo a matriz de confusão
    display_confusion_matrix(t, y, t.unique(), f"{data_type} Confusion Matrix")


# Função principal para carregar dados, treinar o modelo e avaliar
def main():
    # Carregar os dados de treino
    x_train, t_train = load_dataset(
        "breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_train.csv")

    # Carregar os dados de teste
    x_test, t_test = load_dataset(
        "breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_test.csv")

    # Treinar o modelo KNN com os dados de treino
    knn = train_knn(x_train, t_train, n_neighbors=1, p=1)

    # Avaliar o modelo com os dados de treino
    evaluate_model(knn, x_train, t_train, "Training data")

    # Avaliar o modelo com os dados de teste
    evaluate_model(knn, x_test, t_test, "Testing data")


# Chamar a função principal
if __name__ == "__main__":
    main()
