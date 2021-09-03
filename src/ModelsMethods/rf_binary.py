import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

from imblearn.ensemble import BalancedRandomForestClassifier

import pickle


def split_train_test(data_x, data_y, test_size=0.3, save_to_file=True, type=""):
    # Split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=test_size
    )

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # if enabled - save data to file
    if save_to_file:
        np.savetxt(
            "blood-test-thesis/docs/results/X_train_{}.csv".format(type),
            X_train,
            delimiter=",",
        )
        np.savetxt(
            "blood-test-thesis/docs/results/X_test_{}.csv".format(type),
            X_test,
            delimiter=",",
        )
        np.savetxt(
            "blood-test-thesis/docs/results/y_train_{}.csv".format(type),
            y_train,
            delimiter=",",
        )
        np.savetxt(
            "blood-test-thesis/docs/results/y_test_{}.csv".format(type),
            y_test,
            delimiter=",",
        )

    # return train test split
    return X_train, X_test, y_train, y_test


def CreateRandomForestClassifierModel(X_train, X_test, y_train, y_test):
    clf = BalancedRandomForestClassifier(
        random_state=1,
        max_depth=5,
        min_samples_leaf=10,
        min_samples_split=100,
        n_estimators=1200,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print("Confustion Matrix:", confusion_matrix(y_test, y_pred))

    return clf


def CreateRandomForestClassifierGridSearch(X_train, y_train, name="best-grid-params"):
    n_estimators = [100, 300, 500, 800, 1200]
    max_depth = [5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]

    hyperF = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    forest = RandomForestClassifier(random_state=1)

    gridF = GridSearchCV(forest, hyperF, cv=3, verbose=1, n_jobs=-1)
    bestF = gridF.fit(X_train, y_train)

    pickle.dump(bestF, open("{}.sav".format(name), "wb"))

    print(bestF)

    return bestF


def LoadFromFile(path="blood-test-thesis/docs/results/best-grid-params.sav"):
    model = pickle.load(open(path, "rb"))

    print(model.best_params_)

