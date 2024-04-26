import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

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


#Função pra criar a rede
def criar_rede(loss):
    #com as variaveis prontas vamos dar inicio a rede 158 → (316 + 1) / 2
    rna = Sequential()
    rna.add(Dense(units= 158, activation='relu',kernel_initializer = 'random_uniform', input_dim = 316)) #1ª camada escondida + camada input
    rna.add(Dense(units= 158, activation='relu',kernel_initializer = 'random_uniform'))#2ª camada escondida
    rna.add(Dense(units= 158, activation='relu',kernel_initializer = 'random_uniform'))#3ª camada escondida
    rna.add(Dense(units= 1, activation='linear'))#camada de saida
    rna.compile(loss = loss,
                optimizer = 'adam',
                metrics = ['mean_absolute_error'])
    
    return rna


rna = KerasRegressor(build_fn= criar_rede,
                                epochs = 100,
                                batch_size = 300)

#usando o kfold = 10 - cv = 10
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
resultado = cross_val_score(estimator = rna,
                            X = previsores,
                            y = preco_real,
                            cv = 10,
                            scoring = 'neg_mean_absolute_error')

# Não é necessário alterar o parâmetro metrics pois ele é usado somente para 
# mostrar o resultado e de fato ele não é utilizado no treinamento da rede neural

regressor = KerasRegressor(build_fn = criar_rede, epochs = 100, batch_size = 300)
parametros = {'loss': ['mean_squared_error', 'mean_absolute_error',
                       'mean_absolute_percentage_error', 'mean_squared_logarithmic_error',
                       'squared_hinge']}


grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parametros,                           
                           cv = 10)
grid_search = grid_search.fit(previsores, preco_real)
melhores_parametros = grid_search.best_params_         #squared_hinge