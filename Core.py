#Passo 1: importe as bibliotecas necessárias e leia o conjunto de dados de treinamento. Acrescente ambos.

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

train=pd.read_csv('E:/!AULAS_UP/Hackathon/bank-additional-full.csv')
test=pd.read_csv('E:/!AULAS_UP/Hackathon/bank-additional.csv')
train['Type']='Train' #Cria uma bandeira para o conjunto de dados de treino e de teste
test['Type']='Test'
fullData = pd.concat([train,test],axis=0) #Combina ambos conjuntos de dados de treino e de teste

#Passo 2: Passo 2 do framework não é necessário no Python. Para o próximo passo.
#Passo 3: Veja os nomes das colunas e o resumo do conjunto de dados.

fullData.columns # This will show all the column names
fullData.head(10) # Show first 10 records of dataframe
fullData.describe() #You can look at summary of numerical fields by using describe() function

#Passo 4: Identifique as (a) variáveis de ID, as (b) variáveis-alvo, as (c) as variáveis categóricas, as (d) as variáveis numéricas e (e) outras variáveis.

ID_col = ['REF_NO']
target_col = ["Account.Status"]
cat_cols = ['children','age_band','status','occupation','occupation_partner','home_status','family_income','self_employed', 'self_employed_partner','year_last_moved','TVarea','post_code','post_area','gender','region']
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col)-set(data_col))
other_col=['Type'] #Identificador do conjunto de dados de treino e de teste

#Passo 5: Identifique as variáveis com valores faltantes e crie uma bandeira (flag) para eles

fullData.isnull().any() #Vai retornar Falso ou Verdadeiro, Verdadeiro significa que há valor faltante senão Falso

num_cat_cols = num_cols+cat_cols # Combina variáveis numéricas e categóricas

#Cria uma uma variável para cada variável que tenha valores faltantes com VariableName_NA
# e marca o valor faltante com 1 e os outros com 0

for var in num_cat_cols:
   if fullData[var].isnull().any()== True:
      fullData[var+'_NA']=fullData[var].isnull()*1

#Passo 6: impute os valores faltantes

#Preenche valores faltantes com a média
fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].mean(),inplace=True)

#Preenche valores categóricos faltantes com -9999
fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)

#Passo 7: Crie codificadores de etiqueta para as variáveis categóricas e divida o conjunto de dados em treino e teste, divida ainda mais o conjunto de treino em treinamento e validação.

#cria rótulos codificadores para itens categóricos
for var in cat_cols:
 number = LabelEncoder()
 fullData[var] = number.fit_transform(fullData[var].astype('str'))

#Variável alvo também é categórica então é convertida
fullData["Account.Status"] = number.fit_transform(fullData["Account.Status"].astype('str'))

train=fullData[fullData['Type']=='Train']
test=fullData[fullData['Type']=='Test']

train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
Train, Validate = train[train['is_train']==True], train[train['is_train']==False]

#Passo 8: Passe para o processo de modelagem as variáveis imputadas e as dummies (valores faltantes marcados com bandeiras). Estou usando Random Forest para prever a classe.

features=list(set(list(fullData.columns))-set(ID_col)-set(target_col)-set(other_col))

x_train = Train[list(features)].values
y_train = Train["Account.Status"].values
x_validate = Validate[list(features)].values
y_validate = Validate["Account.Status"].values
x_test=test[list(features)].values

random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)

#Passo 9: Verifique o desempenho e faça previsões

status = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, status[:,1])
roc_auc = auc(fpr, tpr)
print(roc_auc)

final_status = rf.predict_proba(x_test)
test["Account.Status"]=final_status[:,1]
test.to_csv('E:/!AULAS_UP/Hackathon/model_output.csv',columns=['REF_NO','Account.Status'])

