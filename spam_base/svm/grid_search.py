import pandas
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param = [
    {
        'C': [0.09, 0.1, 0.2],
        'kernel': ['linear', 'poly', 'sigmoid', 'rbf']
    },
]


gs = GridSearchCV(
    # SVC(),
    # Ajuste para lidar com classes desbalanceadas
    SVC(class_weight='balanced'),
    param,
    # scoring='precision',
    scoring='f1_macro',
    verbose=True
)

dataset = pandas.read_csv(
    "spam_base/pre_processed/spambase_train.csv")
x_train = dataset.drop(columns=['spam'])
t_train = dataset['spam']


# Fit the model
gs.fit(x_train, t_train)

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)
