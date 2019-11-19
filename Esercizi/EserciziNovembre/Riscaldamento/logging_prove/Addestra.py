from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pandas as pd


def addestraSVC(X, y, c, gamma, kernel, dim):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    pca = PCA(n_components=dim)

    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    model = SVC(gamma=gamma, C=c, kernel=kernel)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


def addrestraDecisionTreeClassifier(X, y, criterion, dim):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    pca = PCA(n_components=dim)

    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    model = DecisionTreeClassifier(criterion=criterion)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


def addestraRandomForestClassifier(X, y, n_estimators, dim):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    pca = PCA(n_components=dim)

    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


def addestraLinearDiscriminantAnalysis(X, y, solver, dim):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    pca = PCA(n_components=dim)

    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    model = LinearDiscriminantAnalysis(solver=solver)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


def addestraNaiveBayes(X, y, dim):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30, stratify=y)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    pca = PCA(n_components=dim)

    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)


# funzione che calcola il valore degli iperparametri per il modello dato in input
def trovaIperparametri(X, y, model, numero_dimensioni):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30, stratify=y)

    if model == SVC:
        steps = [
            ('scaler', StandardScaler()),
            ('reduce_dim', PCA()),
            ('SVM', SVC()),
        ]

        pipeline = Pipeline(steps)

        valori_C = np.arange(0.1, 1.0, 0.1)
        valori_gamma = [0.1, 0.01]
        valori_kernel = ['linear', 'poly', 'rbf', 'sigmoid']

        params = {'SVM__C': valori_C,
                  'SVM__gamma': valori_gamma,
                  'SVM__kernel': valori_kernel,
                  'reduce_dim__n_components': np.arange(1, numero_dimensioni, 1),
                  }
    elif model == DecisionTreeClassifier:

        steps = [
            ('scaler', StandardScaler()),
            ('reduce_dim', PCA()),
            ('tree', DecisionTreeClassifier()),
        ]

        pipeline = Pipeline(steps)

        params = {'reduce_dim__n_components': np.arange(1, numero_dimensioni, 1),
                  'tree__criterion': ['gini', 'entropy'],
                  }

    elif model == RandomForestClassifier:

        steps = [
            ('scaler', StandardScaler()),
            ('reduce_dim', PCA()),
            ('random_forest', RandomForestClassifier()),
        ]

        pipeline = Pipeline(steps)

        params = {'reduce_dim__n_components': np.arange(1, numero_dimensioni, 1),
                  'random_forest__n_estimators': np.arange(10, 100, 10),
                  }

    elif model == LinearDiscriminantAnalysis:

        steps = [
            ('scaler', StandardScaler()),
            ('reduce_dim', PCA()),
            ('linear_discriminant_analysis', LinearDiscriminantAnalysis()),
        ]

        pipeline = Pipeline(steps)

        params = {'reduce_dim__n_components': np.arange(1, numero_dimensioni, 1),
                  'linear_discriminant_analysis__solver': ['svd', 'lsqr', 'eigen'],

                  }
    elif model == GaussianNB:
        steps = [
            ('scaler', StandardScaler()),
            ('reduce_dim', PCA()),
            ('linear_discriminant_analysis', LinearDiscriminantAnalysis()),
        ]

        pipeline = Pipeline(steps)

        params = {'reduce_dim__n_components': np.arange(1, numero_dimensioni, 1),

                  }
    grid = GridSearchCV(pipeline, param_grid=params, cv=3)

    grid.fit(X_train, y_train)

    return grid.best_params_

