import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param = [
    { 'n_neighbors': range(1,21), 'p': [1, 2, 3, 4, 5] },
]

gs = GridSearchCV(
    KNeighborsClassifier(),
    param,
    scoring='f1_macro',
    verbose=True
)

dataset = pandas.read_csv("pre_processed/iris_train.csv")
x_train = dataset.drop(columns=['class'])
t_train = dataset['class'] # actual outputs (targets)


# Fit the model
gs.fit(x_train, t_train)

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)