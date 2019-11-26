#Bibliotecas necessarias para ler o conjunto de dados

#Realizar calculos numericos em array
import Induction
#Analise de dados
import pandas as pd
#Efetuar a regressão logistica
from sklearn.externals import joblib
#Codificar os dados de string para regressao
from sklearn.preprocessing import LabelEncoder  # Need to encode string data for regression

#Leitura do arquivo após pré tratamento dos dados em Excel
novo = pd.read_csv("new-instance.csv")
#Tratamento dos dados

#Substitua todos os valores desconhecidos por 0
novo = novo.replace('unknown', 0)


#Cabecalho
novo.head()

#Simplificando o conjunto de dados

#Agrupa educacoes basicas
novo.loc[novo['education'] == 'basic.4y', 'education'] = 'basic'
novo.loc[novo['education'] == 'basic.6y', 'education'] = 'basic'
novo.loc[novo['education'] == 'basic.9y', 'education'] = 'basic'

#Agrupa trabalhos de colarinho branco
novo.loc[novo['job'] == 'admin.', 'job'] = 'white-collar'
novo.loc[novo['job'] == 'management', 'job'] = 'white-collar'
novo.loc[novo['job'] == 'entrepreneur', 'job'] = 'white-collar'
novo.loc[novo['job'] == 'technician', 'job'] = 'white-collar'

# Agrupa trabalhos de colarinho azul e servicos
novo.loc[novo['job'] == 'services', 'job'] = 'blue-collar'
novo.loc[novo['job'] == 'housemaid', 'job'] = 'blue-collar'
novo.loc[novo['job'] == 'services', 'job'] = 'blue-collar'

#Exploracao dos dados
novo.describe()

#Ajustes
#Necessidade de transformar dados de string em valores numericos antes de aplicar o modelo

#Seleciona todas as colunas de dados nao numericos
processado = novo.select_dtypes(exclude=['number'])

#Transforma valores de string em valores numericos
processado = processado.apply(LabelEncoder().fit_transform)
#Associa as colunas recém-codificadas ao restante do quadro
processado = processado.join(novo.select_dtypes(include=['number']))

#Nome do modelo em disco
filename = 'modelo_finalizado.sav'

#Carrega o modelo do disco
modelo_carregado = joblib.load(filename)

#Efetua a previsao
previsao = modelo_carregado.predict_proba(processado)

#Taxa de sucesso
print("Probabilidade de Contratacao:", (previsao[0,1]) * 100, "%")
print("Probabilidade de Não Contratacao:", (previsao[0,0]) * 100, "%")