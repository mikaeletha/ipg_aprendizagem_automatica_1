import pandas
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# c_values = np.logspace(-3, 3, 7)
c_values = np.arange(0, 1.1, 0.1)

param = [
    {'C': c_values,
     'kernel': ['linear', 'poly', 'sigmoid', 'rbf']
     },
]


gs = GridSearchCV(
    # LinearSVC(),
    SVC(),
    param,
    scoring='precision',
    verbose=True
)

dataset = pandas.read_csv(
    "spam_base/pre_processed/spambase_train.csv")
x_train = dataset.drop(columns=['spam'])
t_train = dataset['spam']  # actual outputs (targets)


# Fit the model
gs.fit(x_train, t_train)

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)
