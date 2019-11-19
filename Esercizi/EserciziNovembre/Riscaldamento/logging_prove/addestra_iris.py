from Addestra import *
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import logging


cancer = load_breast_cancer()

Xc = cancer['data']
yc = cancer['target']


diz = trovaIperparametri(X=Xc, y=yc, model=SVC, numero_dimensioni=30)
c = diz['SVM__C']
gamma = diz['SVM__gamma']
kernel = diz['SVM__kernel']
n = diz['reduce_dim__n_components']
score = addestraSVC(Xc, yc, c=c, gamma=gamma, kernel=kernel, dim=n)

logging.basicConfig(filename='prova.log', level=logging.DEBUG, filemode='a', datefmt='%H:%M:%S')
logging.info(f'Usando una SVC con c={c}, gamma={gamma}, kernel={kernel}, numeri di dimensioni {n} sia ha uno score di {score}')


