from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as pd

np.random.seed(2)
datasets = load_iris()
X = datasets['data']
y = datasets['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = SVC(gamma='scale')

model.fit(X_train, y_train)

p = model.predict(X_train)


acc = accuracy_score(y_train, p)

print(f'Score {acc}')
