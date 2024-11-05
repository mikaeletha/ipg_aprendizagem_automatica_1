import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor


def optimize_knn(file_path):
    param = [
        {'n_neighbors': range(1, 40), 'p': [1, 2, 3]},
    ]

    gs = GridSearchCV(
        # KNeighborsClassifier(),
        KNeighborsRegressor(),
        param,
        # scoring='f1_macro',
        scoring='neg_mean_squared_error',
        verbose=True
    )

    dataset = pd.read_csv(file_path)
    x_train = dataset.drop(columns=['quality'])
    t_train = dataset['quality']

    gs.fit(x_train, t_train)

    print(f"Best parameters for {file_path}: ", gs.best_params_)
    print(f"Best score for {file_path}: ", gs.best_score_)


optimize_knn("wine_quality/pre_processed/wine_quality_red_train.csv")
optimize_knn("wine_quality/pre_processed/wine_quality_white_train.csv")
