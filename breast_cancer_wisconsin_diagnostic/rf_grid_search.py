import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param = [
    {
        'n_estimators': [100, 500, 1000],
        'max_features': ["sqrt", "log2", 10, 15]
    },
]

gs = GridSearchCV(
    RandomForestClassifier(),
    param,
    scoring='recall',
    verbose=True
)

output_var = "Diagnosis"

dataset = pandas.read_csv(
    "breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_train.csv")
x_train = dataset.drop(columns=[output_var])
t_train = dataset[output_var]

gs.fit(x_train, t_train)

print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)
