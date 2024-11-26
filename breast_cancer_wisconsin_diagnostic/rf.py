import matplotlib.pyplot as plt
import pandas
from numpy import sort
from sklearn.calibration import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.svm import SVC


def display_confusion_matrix(targets, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(
        targets, predicted, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()


output_var = "Diagnosis"

dataset = pandas.read_csv(
    "breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_train.csv")
x_train = dataset.drop(columns=[output_var])
t_train = dataset[output_var]  # actual outputs (targets)

test_dataset = pandas.read_csv(
    "breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_test.csv")
x_test = test_dataset.drop(columns=[output_var])
t_test = test_dataset[output_var]  # actual outputs (targets)

rf = RandomForestClassifier(
    max_features='sqrt',
    n_estimators=100
)
rf.fit(x_train, t_train)

# model predicted outputs
y_train = rf.predict(x_train)
y_test = rf.predict(x_test)

classes = dataset[output_var].unique()

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

display_confusion_matrix(t_train, y_train, classes,
                         "Training data confusion matrix")
display_confusion_matrix(t_test, y_test, classes, "Test data confusion matrix")
