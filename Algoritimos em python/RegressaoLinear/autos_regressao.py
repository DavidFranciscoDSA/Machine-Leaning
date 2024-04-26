import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from sklearn.compose import ColumnTransformer

#Carregando a base de dados
diretorio = "C:/Users/david/Desktop/ESTUDOS/DeepLearningAaZ/DeepLearning/RegressaoUmValor/"
nomeArquivo = "autos.csv"
base = pd.read_csv(diretorio+nomeArquivo, encoding = 'ISO-8859-1')

#contando um numero de registros de determinada variavel para ver de tem influencia no modelo
base['abtest'].value_counts()

#As variaveis dateCrawled, dateCreated, nrOfPictures, postalCode, lastSeen, name, seller, offerType
#não fazem sentido da analise de preço do veiculo e serão excluidas
colunasDescartadas = ['offerType','dateCrawled', 'dateCreated', 'nrOfPictures', 'postalCode', 'lastSeen','name','seller']
base = base.drop(columns = colunasDescartadas, axis=1)

#excluir valores inconsistentes
#preços menosres que 100 euros - vamos excluir da base
precoInc = base.loc[base.price <=100] #tem 14352
precoInc2 = base.loc[base.price >=350000] #tem 115
base = base[base.price > 100]
base = base[base.price < 350000]

#Substituindo os valores nan pela modas das variaveis - é feito em todas as variaveis
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() #limousine

base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() #manuell

base.loc[pd.isnull(base['model'])]
base['model'].value_counts() #golf

base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() #benzin

base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() #nein

base.loc[pd.isnull(base['price'])]
base['price'].value_counts() #ta ok

#variaveis e valores - dicionario
valores = {'vehicleType' : 'limousine',
           'gearbox' : 'manuell',
           'model' : 'golf',
           'fuelType' : 'benzin',
           'notRepairedDamage':'nein'}

#substituindo na base
base = base.fillna(value = valores)

#acima foram tratados os dados agora vamos separar em treino e teste - preço fica fora
previsores = base.iloc[:,1:13].values
preco_real = base.iloc[:,0].values

#Vamos transformar as variaveis categoricas em numericas 
#['test', 'limousine', 1993, 'manuell', 0, 'golf', 150000, 0,'benzin', 'volkswagen', 'nein']
#   0         1                  3           5                  8           9          10 
lbEnc_previsores = LabelEncoder()
categoricas = [0,1,3,5,8,9,10]
for i in range(len(categoricas)):
    previsores[:,categoricas[i]] = lbEnc_previsores.fit_transform(previsores[:,categoricas[i]])

#vamos transformar essas mesmas variaveis em dummy
column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categoricas)],remainder='passthrough')
previsores = column_transformer.fit_transform(previsores)

#com as variaveis prontas vamos dar inicio a rede 158 → (316 + 1) / 2

rna = Sequential()
rna.add(Dense(units= 158, activation='relu', input_dim = 316)) #1ª camada escondida + camada input
rna.add(Dense(units= 158, activation='relu'))#2ª camada escondida
rna.add(Dense(units= 158, activation='relu'))#3ª camada escondida
rna.add(Dense(units= 1, activation='linear'))#camada de saida
rna.compile(loss = 'mean_absolute_error',
            optimizer = 'adam',
            metrics = ['mean_absolute_error'])
#batch_size = 300 → a cada 300 registros atualiza o peso
rna.fit(previsores, preco_real, batch_size=300, epochs=300)

#obtendo a previsao
previsoes = rna.predict(previsores)









