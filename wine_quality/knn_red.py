import matplotlib.pyplot as plt
import pandas
from numpy import sort
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier


def display_confusion_matrix(targets, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(
        targets, predicted, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()


dataset = pandas.read_csv(
    "wine_quality/pre_processed/wine_quality_red_train.csv")
x_train = dataset.drop(columns=['quality'])
t_train = dataset['quality']

test_dataset = pandas.read_csv(
    "wine_quality/pre_processed/wine_quality_red_test.csv")
x_test = test_dataset.drop(columns=['quality'])
t_test = test_dataset['quality']

# Best parameters for wine_quality/pre_processed/wine_quality_red_train.csv:  {'n_neighbors': 12, 'p': 1}

n_neighbors = 12
p = 1
knn = KNeighborsClassifier(n_neighbors, p=p)
knn.fit(x_train, t_train)

y_train = knn.predict(x_train)
y_test = knn.predict(x_test)

quality = dataset['quality'].unique()

print("Training data:")
print(f"accuracy: {accuracy_score(t_train, y_train) * 100:.2f}%")

print("Testing data:")
print(f"accuracy: {accuracy_score(t_test, y_test) * 100:.2f}%")

print("Train report")
train_report = classification_report(t_train, y_train, digits=4)
print(train_report)

print("Test report")
test_report = classification_report(t_test, y_test, digits=4)
print(test_report)

display_confusion_matrix(t_train, y_train, quality,
                         "Training data confusion matrix")
display_confusion_matrix(t_test, y_test, quality, "Test data confusion matrix")
