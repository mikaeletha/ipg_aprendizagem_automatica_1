import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def optimize_knn(file_path):
    param = [
        {
            'n_neighbors': range(1, 21),
            'p': [1, 2, 3, 4, 5]
        }
    ]

    gs = GridSearchCV(
        KNeighborsClassifier(),
        param,
        scoring='recall_macro',
        verbose=True
    )

    dataset = pd.read_csv(file_path)
    x_train = dataset.drop(columns=['Diagnosis'])
    t_train = dataset['Diagnosis']

    gs.fit(x_train, t_train)

    print(f"Best parameters for {file_path}: ", gs.best_params_)
    print(f"Best score for {file_path}: ", gs.best_score_)


# optimize_knn("breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_train.csv")
optimize_knn("breast_cancer_wisconsin_diagnostic/pre_processed/wdbc_test.csv")
