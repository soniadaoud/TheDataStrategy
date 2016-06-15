

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('data/subventions-accordees-et-refusees.csv',sep=';')

l = int(len(df)/5) #80% of df

train_data = df[:4*l] #first 80% is train set
test_data = df[4*l:] #last 20% is test set 

#prepocessing
df.drop('Année', axis=1, inplace=True)
df.drop('Adresse', axis=1, inplace=True)
df.drop('Nom du partenaire', axis=1, inplace=True)
df.drop('N° SIRET', axis=1, inplace=True)

df['Ville']=((df['Code postal'].values / 1000))

df.drop('Code postal', axis=1, inplace=True)
df.drop('Intitulé de la demande', axis=1, inplace=True)
df.drop('Montant voté par demande', axis=1, inplace=True)
df.drop('Total voté en 2013', axis=1, inplace=True)
df.drop('S-PR-Numéro SIMPA', axis=1, inplace=True)
df['Appel à projets']= (df['Appel à projets']=='O')*1
df['Appel à projets Politique Ville']= (df['Appel à projets Politique Ville']=='O')*1
df['Financée/Non Financée']= (df['Financée/Non Financée']=='O')*1

for column in df:
    df[column].fillna(0,inplace=True)
    

#end of preprocessing 


x_train = df.drop('Financée/Non Financée',axis=1)
y_train = df['Financée/Non Financée']

x_test = df.drop('Financée/Non Financée',axis=1)
y_test = df['Financée/Non Financée']

gnb = GaussianNB()
y_pred = gnb.fit(x_train,y_train).predict(x_test)

print("Number of mislabeled points out of a total %d points : %d"
         % (len(x_test),(y_test != y_pred).sum()))
