#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.linear_model import Lasso, LogisticRegression, RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, auc
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[3]:


file= pd.read_table("C:/Users/hp/pfedossier/fichier_aprés_pre_processing",sep='|')


# In[ ]:


file['Statut'].value_counts()


# In[4]:


file.info()


# In[ ]:


print(file.loc[file['Statut']==1,:])


# In[ ]:


file['Statut']=file['Statut'].replace('bon',0)


# In[ ]:


file['Statut']=file['Statut'].replace('mauvais',1)


# In[ ]:


import csv
file.to_csv('fichier_aprés_pre_processing',sep="|")


# In[ ]:


file['SituationMatrimoniale']=file['SituationMatrimoniale'].replace('N','M', regex=True)


# In[ ]:


file['SituationMatrimoniale']=file['SituationMatrimoniale'].replace('V','C', regex=True)


# In[4]:


numerical = []
for col in file.columns:
    if (file[col].dtype == np.int64 or file[col].dtype == np.float64) and not col=='Statut' and not col=='Unnamed: 0'and not col=='Unnamed: 0.1' and not col=='Unnamed: 0.1.1':
        numerical.append(col)

categorical = []
for col in file.columns:
    if file[col].dtype == np.object and not col=='Client':
        categorical.append(col)

print(numerical)
print(categorical)


# In[5]:


df = pd.DataFrame()
mms = MinMaxScaler()
df[numerical] = pd.DataFrame(mms.fit_transform(file[numerical]))
df = pd.concat([df, pd.get_dummies(file[categorical]), file['Statut']], axis=1)
df.info()
df.head(5)


# In[ ]:


import csv
df.to_csv('test',sep="|")


# In[ ]:


#suppression des outliers
indices = file[file['age'] <18].index
file.drop(indices, inplace=True)


# # Neural network
# 

# In[6]:


X = df.drop('Statut',axis=1)
Y = df.Statut
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)


# In[ ]:


mlp = MLPClassifier()
mlp.fit(X_train3,y_train3)


# In[ ]:


param_grid=[{"hidden_layer_sizes":list([(3,),(4,),(5,),(6,),(7,),(8,),(8,),(10,)])}]
para= GridSearchCV(MLPClassifier(max_iter=700),param_grid,cv=10,n_jobs=-1)
mlp=para.fit(X_train3, y_train3)
# optimal parameter
print("Meilleur score = %f, Meilleur paramètre = %s" % (1. - mlp.best_score_,mlp.best_params_))


# In[7]:


rna = MLPClassifier(hidden_layer_sizes=(4,),activation="logistic",solver="lbfgs")
rna.fit(X_train3,y_train3)


# In[8]:


rna.predict(X_test3) #Pour avoir les prédictions


# In[ ]:


#affichage des coefficients
print(rna.coefs_)


# In[ ]:


print(rna.intercepts_) #les coefficients


# In[ ]:


result_mlp = 1-rna.score(X_test3,y_test3)
print(result_mlp)

y_pred_mlp = rna.predict(X_test3)

resultsTrain.append(1 - rna.score(X_train3,y_train3))
results.append(result_mlp)
gini.append(Gini(y_test3,y_pred_mlp))
results_labels.append('MLP')


# In[ ]:


#prédiction sur l'échantillon test
y_pred = rna.predict(X_test3)
print(y_pred)


# In[ ]:


#figur
fpr, tpr, thresholds = roc_curve(y_test2, y_pred[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


rna = MLPClassifier(hidden_layer_sizes=(4,8),activation="logistic",solver="lbfgs")
rna.fit(X_train3,y_train3)
y_pred = rna.predict(X_test3)
from sklearn import metrics
print(metrics.confusion_matrix(y_test3,y_pred))
print(metrics.accuracy_score(y_test3,y_pred))
print("Taux erreur = " + str(1-metrics.accuracy_score(y_test3,y_pred)))


# In[ ]:


y_pred_proba = rna.predict_proba(X_test2)[::,1]


# In[ ]:


y_pred_proba


# In[ ]:


fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Modèle 3, auc="+str(auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc=4)
plt.show()


# In[ ]:


labels = ['Bons','Mauvais']
classes = pd.value_counts(file['Statut'], sort = True)
classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), labels)
plt.xlabel("prêt")
plt.ylabel("Frequency")


# In[ ]:


plt.scatter(file['Revenus'], file['age'],c='blue', alpha=0.5)
plt.xlabel("Revenus du personne")
plt.ylabel("age")
plt.show()


# In[ ]:


file.hist(column='Revenus',by='Statut',bins=30)


# In[ ]:


ax = sns.countplot('Sexe',hue='Revenus', data = file)
plt.ylabel('Nombre total')
plt.show()


# In[ ]:


import seaborn as sns
sns.set_style('whitegrid')
ax=sns.boxplot(x='Statut',y='Revenus' , data=file)
ax=sns.stripplot(x='Statut',y='Revenus',data=file)


# In[ ]:


gender_df = file.groupby(['Sexe','Statut'])['Revenus'].value_counts()
gender_df


# In[ ]:


#suppression des outliers
indices = file[file['age'] <18].index
file.drop(indices, inplace=True)


# In[ ]:


file.drop(file[ (file['Revenus'] !='de 100,000 a 200,000') & (file['Revenus']!='de 50,000 a 100,000')& (file['Revenus']!='SANS REVENU FIXE') 
                 &(file['Revenus']!='plus de 500,000')& 
                 (file['Revenus'] !='0')& 
                (file['Revenus']!='de 35,000 a 50,000') &(file['Revenus'] !='de 350,000 a 500,000') ].index, inplace=True)


# In[ ]:


import seaborn as sns
plt.figure()
sns.set(font_scale=1)
sns.heatmap(file[numerical].corr(), cmap="Blues", annot=True, fmt=".2f")
plt.show()


# In[9]:


fil= pd.read_excel("C:/Users/hp/pfedossier/fichtest.xlsx")


# In[16]:


numerical = []
for col in fil.columns:
    if (fil[col].dtype == np.int64 or fil[col].dtype == np.float64) and not col=='Statut' and not col=='Unnamed: 0'and not col=='Unnamed: 0.1' and not col=='Unnamed: 0.1.1':
        numerical.append(col)

categorical = []
for col in fil.columns:
    if fil[col].dtype == np.object and not col=='Client':
        categorical.append(col)

print(numerical)
print(categorical)


# In[17]:


fi= pd.DataFrame()
mms = MinMaxScaler()
fi[numerical] = pd.DataFrame(mms.fit_transform(fil[numerical]))
fi = pd.concat([fi, fil[categorical]], axis=1)
fi.info()


# In[18]:


rna.predict(fil)


# In[19]:


pred=rna.predict_proba(fil) 


# In[22]:


#[[BON,Mauvais]]
pred


# 
# # Reg

# In[ ]:


f= pd.DataFrame()
mms = MinMaxScaler()
f[numerical] = pd.DataFrame(mms.fit_transform(file[numerical]))
f = pd.concat([f, file[categorical], file['Statut']], axis=1)
f.info()


# In[ ]:


X = df.drop('Statut',axis=1)
Y = df.Statut
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle=True, random_state=1234)
print(X_train.shape)
print(X_test.shape)


# In[ ]:


LR = LogisticRegression(random_state=1234)
parameters = {'penalty':['l1', 'l2'], 'C':[10, 9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001]}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
clf = GridSearchCV(LR, parameters, scoring='roc_auc', cv=skf)
clf.fit(X_train, y_train)

print('The best parameters for Logistic Regression: ', clf.best_params_)
print('ROC-AUC score on train set: ', round(clf.best_score_, 4))

y_pred1 = clf.predict_proba(X_test)
roc_score = roc_auc_score(y_test, y_pred1[:, 1])
print('ROC-AUC score on test set: ', round(roc_score, 4))


# In[ ]:


clf = LogisticRegression(penalty='l1', C=10, random_state=1234)
clf.fit(X_train,y_train)
result_rg = 1-clf.score(X_test,y_test)
print(result_rg)

y_pred_rg = clf.predict(X_test)

resultsTrain.append(1 - clf.score(X_train,y_train))
results.append(result_rg)
gini.append(Gini(y_test,y_pred_rg))
results_labels.append('RL')


# In[ ]:


Gini(y_test2,y_pred)


# In[ ]:


LR = LogisticRegression(penalty='l1', C=10, random_state=1234)
LR.fit(X_train,y_train)
print(LR.coef_)
print(LR.intercept_)


# In[ ]:


import sklearn.metrics as metrics
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Modèle 1, auc="+str(auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc=4)
plt.show()


# In[ ]:


LR = LogisticRegression(penalty='l1', C=10, random_state=1234)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
selector = RFECV(estimator=LR, step=1, cv=skf, scoring='roc_auc')
selector.fit(X_train, y_train)

print('The optimal number of features is {}'.format(selector.n_features_))
features = [f for f,s in zip(X_train.columns, selector.support_) if s]
print('The selected features are:')
print ('{}'.format(features))


# In[ ]:





# In[ ]:


d= pd.DataFrame()
d = df[['NbJoursCR', 'NbJoursDB', 'nbp', 'age', 'SECTEURA_PARTICULIERS SALARIES', 'Sexe_M', 'Revenus_0', 'Revenus_SANS REVENU FIXE', 'Revenus_de 100,000 a 200,000', 'Revenus_de 350,000 a 500,000', 'Revenus_de 50,000 a 100,000', 'Revenus_plus de 500,000']]
d= pd.concat([d, file['Statut']], axis=1)
d.info()


# In[ ]:


clf=LogisticRegression(penalty='l1', C=10, random_state=1234)
clf.fit(X_train, y_train)
# Decreasing features importance
importances_log = abs(clf.coef_[0])
importances_log = importances_log/np.sum(importances_log)
indices= np.argsort(importances_log)[::-1]
for f in range(X.shape[1]):
    print(df.columns[indices[f]], importances_log[indices[f]])


# In[ ]:


plt.figure(figsize=(10,15))
plt.title("Importances des variables")
plt.barh(range(X_train.shape[1]), importances_log[indices][::-1], align='center')
plt.yticks(range(X_train.shape[1]), [df.columns[i] for i in indices[::-1]])
plt.show()


# In[ ]:


clf=LogisticRegression(penalty='l1', C=0.8, random_state=1234)
clf.fit(X_train3, y_train3)

y_pred3 = clf.predict_proba(X_test3)
roc_score = roc_auc_score(y_test3, y_pred3[:, 1])
print('ROC-AUC score on test set: ', round(roc_score, 4))


# In[ ]:


from sklearn import metrics
print(metrics.confusion_matrix(y_test2,y_pred))
print(metrics.accuracy_score(y_test2,y_pred))
print("Taux erreur = " + str(1-metrics.accuracy_score(y_test2,y_pred)))


# In[ ]:


X = d.drop('Statut',axis=1)
Y = d.Statut
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=1234)


# In[ ]:


import statsmodels.api as sm
logit_model=sm.Logit(y_train3,X_train3)
result=logit_model.fit()
print(result.summary())


# In[ ]:


y_pred3 = result.predict(X_test3)
roc_score = roc_auc_score(y_test3, y_pred3)
print('ROC-AUC score on test set: ', round(roc_score, 4))


# In[ ]:


y_pred3


# In[ ]:


pos = pd.get_dummies(y_test).as_matrix()


# In[ ]:


pos


# In[ ]:


reg_1 = linear_model.LogisticRegression()
reg_1.fit(X_train, y_train)
#Coefficients:
print( "Coefficients are: {0}".format(reg_1.coef_))

#Intercept is:
print ("Intercept is: {0}".format(reg_1.intercept_))


# In[ ]:


plt.figure(figsize=(15,5))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (ROC-AUC score)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.savefig('feature_auc_nselected.png', bbox_inches='tight', pad_inches=1)


# In[ ]:


selected = selector.estimator_.coef_.reshape(-1, 1)
table_rfecv = pd.DataFrame(selected, columns = ['coeff']) 
table_rfecv['feature'] = features
table_rfecv.sort_values('coeff', ascending=False)


# # Decision Tree

# In[ ]:


df_tree = pd.DataFrame()
df_tree[numerical] = file[numerical]
df_tree = pd.concat([df_tree, pd.get_dummies(file[categorical]), file['Statut']], axis=1)
df_tree.info()

X = df_tree.drop('Statut',axis=1)
Y = df_tree.Statut
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1234)

df_tree.head(5)


# In[ ]:


print(X_train2.shape)


# In[ ]:


parameters = {'criterion': ['gini', 'entropy'], 'max_depth':[x for x in range(3, 20)], 'min_samples_leaf': [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 500]}
dtc = DecisionTreeClassifier(random_state=1234)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
cl = GridSearchCV(dtc, parameters, scoring='roc_auc', cv=skf)
cl.fit(X_train2, y_train2)

print('The best parameters for Decision Tree: ', cl.best_params_)
print('ROC-AUC score on training set: ', round(cl.best_score_, 4))

y_pred = cl.predict_proba(X_test2)
roc_score = roc_auc_score(y_test2, y_pred[:, 1])
print('ROC-AUC score on test set: ', round(roc_score, 4))


# In[ ]:


import sklearn.metrics as metrics
y_pred = cl.predict_proba(X_test2)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test2,  y_pred)
auc = metrics.roc_auc_score(y_test2, y_pred)
plt.plot(fpr,tpr,label="Modèle 2, auc="+str(auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc=4)
plt.show()


# In[ ]:


y_pred = clf.predict(X_test1)
table_scores_total['Precision'][1] = round(precision_score(y_test1, y_pred), 2)
table_scores_total['Recall'][1] = round(recall_score(y_test1, y_pred), 2)
table_scores_total['Gini'][1] = round(Gini(y_test1, y_pred), 2)
table_scores_total


# In[ ]:


rf = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=50)
cl = rf.fit(X_train2,y_train2)
y_pred = cl.predict(X_test2)
print(confusion_matrix(y_test2,y_pred))


# In[ ]:


rf = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=50,random_state=1234)
cl = rf.fit(X_train2,y_train2)
y_pred = cl.predict(X_test2)
from sklearn import metrics
print(metrics.confusion_matrix(y_test2,y_pred))
print(metrics.accuracy_score(y_test2,y_pred))
print("Taux erreur = " + str(1-metrics.accuracy_score(y_test2,y_pred)))


# In[ ]:


importances_DT = cl.feature_importances_.round(4)
indices_DT = np.argsort(importances_DT)[::-1]
indices= np.argsort(importances_DT)[::-1]
for f in range(X.shape[1]):
    print(df.columns[indices[f]], importances_DT[indices[f]])


# In[ ]:


plt.figure(figsize=(10,15))
plt.title("Importances des variables")
plt.barh(range(X_train2.shape[1]), importances_DT[indices][::-1], align='center')
plt.yticks(range(X_train2.shape[1]), [df.columns[i] for i in indices[::-1]])
plt.show()


# In[ ]:


Gini(y_test2,y_pred)


# In[ ]:


rf = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_leaf=50,random_state=1234)
# learning
cl = rf.fit(X_train2,y_train2)
result_DT= 1-cl.score(X_test2,y_test2)
print(result_DT)

y_pred_DT = cl.predict(X_test2)

resultsTrain.append(1 - cl.score(X_train2,y_train2))
results.append(result_DT)
gini.append(Gini(y_test2,y_pred_DT))
results_labels.append('DT')


# In[ ]:


dr=pd.DataFrame({'NbJoursCR': [0,10,25,95,32,80],'NbJoursDB':[89,20,56,85,23,25],'nbp':[2,6,5,9,24,52], 'age':[30,53,98,63,40,45],
                                                              })


# In[ ]:


for col in dr.columns:
    if (dr[col].dtype == np.int64 or dr[col].dtype == np.float64) and not col=='Statut' and not col=='Unnamed: 0'and not col=='Unnamed: 0.1' and not col=='Unnamed: 0.1.1':
        numerical.append(col)

categorical = []
for col in dr.columns:
    if dr[col].dtype == np.object and not col=='Client':
        categorical.append(col)

print(numerical)
print(categorical)


# In[ ]:


import csv

dr.to_csv('fich',sep="|")


# In[ ]:


fo= pd.DataFrame()
mms = MinMaxScaler()
fo[numerical] = pd.DataFrame(mms.fit_transform(dr[numerical]))
fo = pd.concat([fo, dr[categorical]], axis=1)
fo.info()


# In[ ]:


y_pred3 = result.predict(fo)


# In[ ]:


y_pred3


# In[ ]:


y_pred = rna.predict(fo)


# In[ ]:


y_pred


# In[ ]:


resultsTrain = []
results = []
gini = []
results_labels = []   


# In[ ]:


def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true


# In[ ]:


plt.figure()
plt.plot(results_labels, results, linestyle = 'dashed', marker = 'o')
plt.xlabel("Méthodes de ML")
plt.ylabel("Erreur")
plt.title("Erreurs obtenues pour les différentes méthodes de ML")
plt.ylim(bottom = 0)
plt.show()


# In[ ]:


plt.figure()
plt.plot(results_labels, resultsTrain, linestyle = 'dashed', marker = 'o')
plt.xlabel("Méthodes de ML")
plt.ylabel("Erreur")
plt.title("Erreurs obtenues pour les différentes méthodes de ML")
plt.ylim(bottom = 0)
plt.show()


# In[ ]:


plt.figure()
plt.plot(results_labels, gini, linestyle = 'dashed', marker = 'o')
plt.xlabel("Méthodes de ML")
plt.ylabel("Indice de Gini")
plt.title("Indices de Gini obtenus pour les différentes méthodes de ML")
plt.ylim(bottom = 0, top = 1)
plt.show()


# In[ ]:


print(gini)

