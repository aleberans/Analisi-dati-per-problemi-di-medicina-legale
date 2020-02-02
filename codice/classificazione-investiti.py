#!/usr/bin/env python
# coding: utf-8

# In[139]:


import logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(filename='Punteggi.log')])
totali = logging.getLogger('Dataset_totali')
totali_with_data = logging.getLogger('Dataset_with_data')
totali_with_bmi = logging.getLogger('Dataset_with_bmi')
totali_with_data_and_bmi = logging.getLogger('Dataset_with_data_and_bmi')
details = logging.getLogger('Details')
details_pca = logging.getLogger('Details_with_pca')
details_tsne = logging.getLogger('Details_with_tsne')


# In[138]:


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


# In[2]:


import pandas as pd
dataset = pd.read_excel('IncidentiModificato.xlsx', sheet_name='Foglio1', index_col=0)
dataset.head()


# In[3]:


len(dataset)


# In[4]:


dataset.index.unique()


# In[5]:


def checkDataset(dataset):
    #controllo che i verbali siano valori unici
    if (len(dataset.index) != len(dataset.index.unique())):
        raise Exception('I verbali non possono essere usati come indice')

    #controllo i valori degli anni
    anni = dataset['ANNI']
    for anno in anni:
        if anno < 1 or anno > 95:
            raise Exception(f'Anno inserito non valido {anno}')
    
    #controllo i valori dei pesi
    pesi = dataset['PESO']
    for peso in pesi:
        if peso < 30 or peso > 120:
            raise Exception(f'Peso inserito non valido {peso}')
    
    #controllo i valori dell'altezza
    altezze = dataset['ALTEZZA']
    for altezza in altezze:
        if altezza < 1.00 or altezza > 2.10:
            raise Exception(f'Altezza inserito non valida {altezza}')
    
    #controllo del BMI
    valori_BMI = dataset['BMI']
    for bmi in valori_BMI:
        if bmi < 10.0 or bmi > 50.0:
            raise Exception(f'bmi inserito non valido {bmi}')
            
    #controllo altri valori compresi tra 0 e 4
    dataset_valori_0_4 = dataset['Testa:Neurocranio']
    for valore in dataset_valori_0_4:
        if valore < 0 or valore > 4:
            raise Exception(f'{valore} non compresa tra 0 e 4')
            

    print("Valori del dataset corretti")
    

checkDataset(dataset)


# In[6]:


import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score
import time
from random import seed
import numpy as np


def addestra(model_class, X, y, model_selection_grid, num_fold_grid_search, num_fold_cross_val, scaling=StandardScaler(), dim_reduction=None):
    
    start_time = time.time()
    
    X_std = scaling.fit_transform(X) if scaling is not None else X
    
    X_std = dim_reduction.fit_transform(X_std) if dim_reduction is not None else X_std
    np.random.seed(42)
    clf = GridSearchCV(estimator=model_class(), param_grid=model_selection_grid, cv=num_fold_grid_search, iid=True, n_jobs=-1)
    val = cross_val_score(clf, X_std, y, cv=num_fold_cross_val)
    print("--- %s seconds ---" % (time.time() - start_time))
    return val


# ## Iniziamo considerando solo i totali per distretto

# In[7]:


X_total = dataset[['SESSO', 'ANNI', 'PESO', 'ALTEZZA', 'Tot Testa', 'Tot Torace', 'Tot Addome', 'Tot Scheletro']]
y_total = dataset['Mezzo']


# In[8]:


risultati = {}


# In[9]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

models = [SVC, DecisionTreeClassifier, RandomForestClassifier, GaussianNB, LinearDiscriminantAnalysis, MLPClassifier]
model_to_string = {SVC: 'SVC', DecisionTreeClassifier: 'DT', RandomForestClassifier:'RF', GaussianNB: 'NB', LinearDiscriminantAnalysis: 'LD', MLPClassifier: 'MLP'}

c_space = np.logspace(-4, 3, 10)
gamma_space = np.logspace(-4, 3, 10)

model_selection_grid_SVC = [
  {'C': c_space, 'kernel': ['linear'], 'gamma': ['auto']},
  {'C': c_space, 'gamma': gamma_space, 'kernel': ['rbf']},
  {'C': c_space, 'gamma': ['auto', 'scale'], 'kernel': ['rbf']},
  {'C': c_space, 'degree': [2, 3, 5, 9], 'kernel': ['poly'], 'gamma': ['auto']},
 ]

model_selection_grid_DT = {'criterion': ['gini', 'entropy'],
                        'max_leaf_nodes': [None, 2, 5, 10, 50, 100],
                        'max_features': [None, 'sqrt', 'log2'],
                        'max_depth': [None, 2, 5, 10]}



model_selection_grid_RF = {'n_estimators': [5, 10, 50, 100, 200],
                        'criterion': ['gini', 'entropy'],
                        'max_leaf_nodes': [None, 2, 5, 10, 50, 100],
                        'max_features': [None, 'sqrt', 'log2'],
                        'max_depth': [None, 2, 5, 10]}

model_selection_grid_NB = {}
model_selection_grid_LD = {}

model_selection_grid_MLP = {'max_iter': [5000],
                        'hidden_layer_sizes': [[2], [4], [6], [10], [20], [4, 4], [10, 10]],
                        'activation': ['identity', 'logistic', 'tanh', 'relu']}

grids = [model_selection_grid_SVC, model_selection_grid_DT, model_selection_grid_RF, model_selection_grid_NB, model_selection_grid_LD, model_selection_grid_MLP]


# In[10]:


risultati['Totali'] = {model_to_string[m]: np.mean(addestra(m, X_total, y_total, g, 9, 9)) for m, g in zip(models, grids)}
totali.info(risultati['Totali'])


# In[11]:


table = pd.DataFrame(risultati)
table


# # Proviamo considerando anche il BMI

# In[12]:


X_total_with_BMI = dataset[['SESSO', 'ANNI', 'PESO', 'ALTEZZA', 'BMI', 'Tot Testa', 'Tot Torace', 'Tot Addome', 'Tot Scheletro']]
y_total_with_BMI = dataset['Mezzo']


# In[13]:


risultati['Totali_with_BMI'] = {model_to_string[m]: np.mean(addestra(m, X_total_with_BMI, y_total_with_BMI, g, 9, 9)) for m, g in zip(models, grids)}
totali_with_bmi.info(risultati['Totali_with_BMI'])


# In[14]:


table = pd.DataFrame(risultati)
table


# # Usiamo la DATA senza BMI

# In[15]:


date_ordinate = dataset['DATA'].sort_values()
prima_data = date_ordinate.values[0]
print("La prima data del dataset è: ", prima_data)


# In[16]:


import datetime as dt

dataset.DATA = dataset.DATA.apply(lambda d: (d - dt.datetime(1970,1,1)).days)

dataset.head()


# In[17]:


X_total_with_data = dataset[['DATA', 'SESSO', 'ANNI', 'PESO', 'ALTEZZA', 'Tot Testa', 'Tot Torace', 'Tot Addome', 'Tot Scheletro']]
y_total_with_data = dataset['Mezzo']


# In[18]:


risultati['Totali_with_data'] = {model_to_string[m]: np.mean(addestra(m, X_total_with_data, y_total_with_data, g, 9, 9)) for m, g in zip(models, grids)}
totali_with_data.info(risultati['Totali_with_data'])


# In[19]:


table = pd.DataFrame(risultati)
table


# # Usiamo ora sia il bmi che la data

# In[20]:


X_total_with_data_and_BMI = dataset[['DATA', 'SESSO', 'ANNI', 'PESO', 'ALTEZZA', 'BMI', 'Tot Testa', 'Tot Torace', 'Tot Addome', 'Tot Scheletro']]
y_total_with_data_and_BMI = dataset['Mezzo']


# In[21]:


risultati['Totali_with_date_and_bmi'] = {model_to_string[m]: np.mean(addestra(m, X_total_with_data_and_BMI, y_total_with_data_and_BMI, g, 9, 9)) for m, g in zip(models, grids)}
totali_with_data_and_bmi.info(risultati['Totali_with_date_and_bmi'])


# In[22]:


table = pd.DataFrame(risultati)
table


# ## Aumentiamo la precisione, scendendo nei cinque livelli di ogni distretto

# In[23]:


selected_cols = ['Testa:Neurocranio',
                 'Testa:Splancnocranio',
                 'Testa:Telencefalo',
                 'Testa:Cervelletto',
                 'Testa:Tronco-encefalico',
                 'Torace:Polmoni',
                 'Torace:Trachea/bronchi',
                 'Torace:Cuore',
                 'Torace:Aorta-toracica',
                 'Torace:Diaframma',
                 'Addome:Fegato',
                 'Addome:Milza',
                 'Addome:Aorta-addominale',
                 'Addome:Reni',
                 'Addome:Mesentere',
                 'Scheletro:Rachide-cervicale',
                 'Scheletro:Rachide-toracico',
                 'Scheletro:Rachide-lombare',
                 'Scheletro:Bacino-e-sacro',
                 'Scheletro:Complesso-sterno/claveo/costale']

X_details = dataset[selected_cols]
y_details = dataset['Mezzo']


# Iniziamo facendo un controllo su quanto sia possibile ridurre la dimensione dei dati

# In[24]:


pca = PCA(n_components = 20)
pca.fit(X_details)


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import pyplot as plt

plt.plot(range(20), pca.explained_variance_ratio_.cumsum())
plt.show()


# In[26]:


pca.explained_variance_ratio_.cumsum()


# In[27]:


pca.explained_variance_ratio_.cumsum()[12]


# Scendendo quindi da 20 a 13 feature si mantiene più del 90% della varianza

# ### Iniziamo comunque con tutte le 20 feature

# In[28]:


risultati['Details'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9)) for m, g in zip(models, grids)}
details.info(risultati['Details'])


# In[29]:


table = pd.DataFrame(risultati)
table


# ### Scendiamo a 13 feature con PCA

# In[30]:


risultati['Details_reduce_PCA'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9, dim_reduction=PCA(n_components=13))) for m, g in zip(models, grids)}
details_pca.info(risultati['Details_reduce_PCA'])


# In[31]:


table = pd.DataFrame(risultati)
table


# ### Proviamo a ridurre le componenti a 10 con t-SNE

# In[32]:


from sklearn.manifold import TSNE


# In[33]:


risultati['Details_reduced_TSNE'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9, dim_reduction=TSNE(n_components=13, method='exact'))) for m, g in zip(models, grids)}
details_tsne.info(risultati['Details_reduced_TSNE'])


# # Usando come scaler StandardScaler

# In[34]:


table = pd.DataFrame(risultati)
table


# # Usando come scaler MinMaxScaler

# In[35]:


from sklearn.preprocessing import MinMaxScaler

risultati_minMax_scaler = {}
risultati_minMax_scaler['Totali'] = {model_to_string[m]: np.mean(addestra(m, X_total, y_total, g, 9, 9, scaling=MinMaxScaler())) for m, g in zip(models, grids)}


# In[36]:


risultati_minMax_scaler['Totali_with_BMI'] = {model_to_string[m]: np.mean(addestra(m, X_total_with_BMI, y_total_with_BMI, g, 9, 9, scaling=MinMaxScaler())) for m, g in zip(models, grids)}


# In[37]:


risultati_minMax_scaler['Totali_with_DATA'] = {model_to_string[m]: np.mean(addestra(m, X_total_with_data,y_total_with_data, g, 9, 9, scaling=MinMaxScaler())) for m, g in zip(models, grids)}


# In[38]:


risultati_minMax_scaler['Totali_with_BMI_and_DATA'] = {model_to_string[m]: np.mean(addestra(m, X_total_with_data_and_BMI,y_total_with_data_and_BMI, g, 9, 9, scaling=MinMaxScaler())) for m, g in zip(models, grids)}


# In[39]:


risultati_minMax_scaler['Details'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9, scaling=MinMaxScaler())) for m, g in zip(models, grids)}


# In[40]:


risultati_minMax_scaler['Details_reduce_PCA'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9, scaling=MinMaxScaler(), dim_reduction=PCA(n_components=13))) for m, g in zip(models, grids)}


# In[41]:


risultati_minMax_scaler['Details_reduced_TSNE'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9, scaling=MinMaxScaler(), dim_reduction=TSNE(n_components=13, method='exact'))) for m, g in zip(models, grids)}


# In[42]:


table2 = pd.DataFrame(risultati_minMax_scaler)
table2


# In[43]:


print("Risultati usando StandardScaler")
display(table)
print("Risultati usando MinMaxScaler")
display(table2)


# # Usando come scaler RobustScaler

# In[44]:


from sklearn.preprocessing import RobustScaler
import pandas as pd


# In[45]:


risultati_RobustScaler = {}
risultati_RobustScaler['Totali'] = {model_to_string[m]: np.mean(addestra(m, X_total, y_total, g, 9, 9, scaling=RobustScaler())) for m, g in zip(models, grids)}


# In[46]:


risultati_RobustScaler['Totali_with_BMI'] = {model_to_string[m]: np.mean(addestra(m, X_total_with_BMI, y_total_with_BMI, g, 9, 9, scaling=RobustScaler())) for m, g in zip(models, grids)}


# In[48]:


risultati_RobustScaler['Totali_with_DATA'] = {model_to_string[m]: np.mean(addestra(m, X_total_with_BMI, y_total_with_BMI, g, 9, 9, scaling=RobustScaler())) for m, g in zip(models, grids)}


# In[49]:


risultati_RobustScaler['Totali_with_BMI_and_DATA'] = {model_to_string[m]: np.mean(addestra(m, X_total_with_data_and_BMI,y_total_with_data_and_BMI, g, 9, 9, scaling=RobustScaler())) for m, g in zip(models, grids)}


# In[50]:


risultati_RobustScaler['Details'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9, scaling=RobustScaler())) for m, g in zip(models, grids)}


# In[51]:


risultati_RobustScaler['Details_reduce_PCA'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9, scaling=RobustScaler(), dim_reduction=PCA(n_components=13))) for m, g in zip(models, grids)}


# In[52]:


from sklearn.manifold import TSNE

risultati_RobustScaler['Details_reduced_TSNE'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9, scaling=RobustScaler(), dim_reduction=TSNE(n_components=13, method='exact'))) for m, g in zip(models, grids)}


# In[53]:


table3 = pd.DataFrame(risultati_RobustScaler)
table3


# In[55]:


print("Risultati usando StandardScaler")
display(table)
print("Risultati usando MaxAbsScaler")
display(table2)
print("Risultati usando RobustScaler")
display(table3)


# # Proviamo ora a non usare uno SCALER

# In[68]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

models2 = [DecisionTreeClassifier, RandomForestClassifier, GaussianNB, LinearDiscriminantAnalysis, MLPClassifier]
model_to_string2 = {DecisionTreeClassifier: 'DT', RandomForestClassifier:'RF', GaussianNB: 'NB', LinearDiscriminantAnalysis: 'LD', MLPClassifier: 'MLP'}

c_space = np.logspace(-4, 3, 10)
gamma_space = np.logspace(-4, 3, 10)

model_selection_grid_SVC = [
  {'C': c_space, 'kernel': ['linear'], 'gamma': ['auto']},
  {'C': c_space, 'gamma': gamma_space, 'kernel': ['rbf']},
  {'C': c_space, 'gamma': ['auto', 'scale'], 'kernel': ['rbf']},
  {'C': c_space, 'degree': [2, 3, 5, 9], 'kernel': ['poly'], 'gamma': ['auto']},
 ]

model_selection_grid_DT = {'criterion': ['gini', 'entropy'],
                        'max_leaf_nodes': [None, 2, 5, 10, 50, 100],
                        'max_features': [None, 'sqrt', 'log2'],
                        'max_depth': [None, 2, 5, 10]}



model_selection_grid_RF = {'n_estimators': [5, 10, 50, 100, 200],
                        'criterion': ['gini', 'entropy'],
                        'max_leaf_nodes': [None, 2, 5, 10, 50, 100],
                        'max_features': [None, 'sqrt', 'log2'],
                        'max_depth': [None, 2, 5, 10]}

model_selection_grid_NB = {}
model_selection_grid_LD = {}

model_selection_grid_MLP = {'max_iter': [5000],
                        'hidden_layer_sizes': [[2], [4], [6], [10], [20], [4, 4], [10, 10]],
                        'activation': ['identity', 'logistic', 'tanh', 'relu']}

grids2 = [model_selection_grid_DT, model_selection_grid_RF, model_selection_grid_NB, model_selection_grid_LD, model_selection_grid_MLP]


# In[73]:


risultati_senza_scaler = {}


# In[74]:


risultati_senza_scaler['Details_reduce_PCA'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9, scaling=None, dim_reduction=PCA(n_components=13))) for m, g in zip(models, grids)}


# In[76]:


risultati_senza_scaler['Details_reduced_TSNE'] = {model_to_string[m]: np.mean(addestra(m, X_details, y_details, g, 9, 9, scaling=None, dim_reduction=TSNE(n_components=13, method='exact'))) for m, g in zip(models, grids)}


# In[77]:


table_senza_scaler = pd.DataFrame(risultati_senza_scaler)
table_senza_scaler


# # Analizziamo i risultati calcolati usando i diversi scaler

# ### Usando StandardScaler

# In[91]:


table.describe()


# In[96]:


table.max()


# ### Usando MinMaxScaler

# In[87]:


table2.describe()


# In[97]:


table2.max()


# ### Usando RobustScaler

# In[89]:


table3.describe()


# In[98]:


table3.max()


# ### Senza usare uno scaler e riducendo la dimensionalità con PCA e TSNE usando il dataset con piu dettaglio

# In[93]:


table_senza_scaler.describe()


# In[99]:


table_senza_scaler.max()


# In generale si puo notare come usando il dataset con 5 zone per distretto sia con riduzione della dimensionalità che senza le prestazioni calino.
# 

# In[ ]:




