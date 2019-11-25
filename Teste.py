#Bibliotecas necessarias para ler o conjunto de dados de treinamento

#Realizar calculos numericos em array
import numpy as np
#Analise de dados
import pandas as pd
#Efetuar a regressão logistica
from sklearn.linear_model import LogisticRegression
#Gerar a matriz de confusao e o relatório de classificacao
from sklearn.metrics import confusion_matrix, classification_report
#Dividir dados
from sklearn.model_selection import train_test_split
#Codificar os dados de string para regressao
from sklearn.preprocessing import LabelEncoder  # Need to encode string data for regression

#Leitura do arquivo após pré tratamento dos dados em Excel
banco = pd.read_csv("E:/!AULAS_UP/Hackathon/bank-full.csv")
inp = pd.read_csv("E:/!AULAS_UP/Hackathon/input_data.csv")

#Tratamento dos dados

#Substitua todos os valores desconhecidos por NaN
banco = banco.replace('unknown', np.NaN)
inp = inp.replace('unknown', np.NaN)

#Apagar todas as linhas com NaN
banco = banco.dropna(axis=0)
inp = inp.dropna(axis=0)

#Imprime dimensoes
print("Dimensoes:", banco.shape, '\n')
#Imprime as caracteristicas
print("Características:", banco.columns.values)

#Cabecalho
banco.head()

#Simplificando o conjunto de dados

# Agrupando educacoes basicas
banco.loc[banco['education'] == 'basic.4y', 'education'] = 'basic'
banco.loc[banco['education'] == 'basic.6y', 'education'] = 'basic'
banco.loc[banco['education'] == 'basic.9y', 'education'] = 'basic'

# Agrupando trabalhos de colarinho branco
banco.loc[banco['job'] == 'admin.', 'job'] = 'white-collar'
banco.loc[banco['job'] == 'management', 'job'] = 'white-collar'
banco.loc[banco['job'] == 'entrepreneur', 'job'] = 'white-collar'
banco.loc[banco['job'] == 'technician', 'job'] = 'white-collar'

# Agrupando trabalhos de colarinho azul e servicos
banco.loc[banco['job'] == 'services', 'job'] = 'blue-collar'
banco.loc[banco['job'] == 'housemaid', 'job'] = 'blue-collar'
banco.loc[banco['job'] == 'services', 'job'] = 'blue-collar'

# Exploração dos dados
banco.describe()

# Histograma dos valores y
pd.DataFrame.hist(banco, column='y', bins=10)
cont = 0
for i in banco['y']:
    if i == 1:
        cont += 1

# Proporcao de  valor 1
print("Proporcao de adesao:", cont / len(banco['y']))
# Proporcao de  valor 0
print("Proporcao de nao adesao:", 1 - cont / len(banco['y']))

#Assinaturas por mes
meses = {}
for i, j in zip(banco['month'], banco['y']):
    if i not in meses:
        meses[i] = j
    else:
        meses[i] += j
meses = pd.Series(meses)
meses.plot.bar(grid=True)

print("meses", meses.sort_values())


#Assinaturas por emprego
empregos = {}
for i, j in zip(banco['job'], banco['y']):
    if i not in empregos:
        empregos[i] = j
    else:
        empregos[i] += j
empregos = pd.Series(empregos)
empregos.plot.bar(grid=True)

print("empregos", empregos.sort_values())

#Assinaturas por idade
idades = {}
for i, j in zip(banco['age'], banco['y']):
    if i not in idades:
        idades[i] = j
    else:
        idades[i] += j
idades = pd.Series(idades)
idades.plot()

print("idades", idades.sort_values())

#Assinaturas por educacao
educacao = {}
for i, j in zip(banco['education'], banco['y']):
    if i not in educacao:
        educacao[i] = j
    else:
        educacao[i] += j
educacao = pd.Series(educacao)
educacao.plot.bar(grid=True)

print("educacao", educacao.sort_values())

# Ajuste do modelo de regressão logistica
# Necessidade de transformar dados de string em valores numericos antes de aplicar o modelo

# Seleciona todas as colunas de dados nao numericos
transformado = banco.select_dtypes(exclude=['number'])
# Transforma valores de string em valores numericos
transformado = transformado.apply(LabelEncoder().fit_transform)
# Associa as colunas recém-codificadas ao restante do quadro
transformado = transformado.join(banco.select_dtypes(include=['number']))

# Dados divididos, 30% para teste
x_treino, x_teste = train_test_split(transformado, test_size = 0.3)
# Isola os valores y para treinamento e teste
y_treino, y_teste = x_treino['y'], x_teste['y']
# Apaga os valores y dos dados de treinamento de entrada
x_treino, x_teste = x_treino.drop(['y'], axis=1), x_teste.drop(['y'], axis=1)

# Ajusta um modelo aos dados de treinamento
modelo = LogisticRegression(solver='lbfgs', multi_class='auto').fit(x_treino, y_treino)

# Previsao dos valores
previsao = modelo.predict(x_teste)

cont = 0
# Compara os valores previstos com os valores reais e conta os erros
for i, j in zip(previsao, y_teste):
    if i == j:
        cont += 1
# Taxa de sucesso
print("Acuracia:", (cont / len(y_teste)) * 100, "%")

# Matriz de confusao e relatorio de classificacao
print(confusion_matrix(y_teste, previsao))
print(classification_report(y_teste, previsao, target_names=['0', '1']))

#Dados tratados
#banco.to_csv('E:/!AULAS_UP/Hackathon/model_output.csv')