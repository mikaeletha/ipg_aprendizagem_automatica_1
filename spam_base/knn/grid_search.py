import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def optimize_knn(file_path):
    param = [
        {
            'n_neighbors': range(1, 3),
            'p': [1, 2, 3]
        }
    ]

    gs = GridSearchCV(
        KNeighborsClassifier(),
        param,
        scoring='precision',
        verbose=True
    )

    dataset = pd.read_csv(file_path)
    x_train = dataset.drop(columns=['spam'])
    t_train = dataset['spam']

    gs.fit(x_train, t_train)

    # Melhor parâmetro encontrado
    print(f"Best parameters: ", gs.best_params_)
    print(f"Best score: ", gs.best_score_)


optimize_knn("spam_base/pre_processed/spambase_test.csv")
