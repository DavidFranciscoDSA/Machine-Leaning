# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:46:22 2024

@author: david
"""

#Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

#Carregando a base de dados
diretorio = "C:/Users/david/Desktop/ESTUDOS/DeepLearningAaZ/DeepLearning/MapaAutoOrganizavel/"
nomeArquivo = "credit_data.csv"
base = pd.read_csv(diretorio+nomeArquivo, encoding = 'ISO-8859-1').dropna()

# Verificar valores menores que zero em todo o DataFrame
negativos_values = (base < 0).sum()
print(negativos_values)
# Verificar valores maiores que 120
fora_values = (base['age'] > 120).sum()
print(fora_values)

#Tratando as idades menores que 0 - substituiremos pela média
media = int(np.mean(base.age)//1)
base.loc[base.age < 0, 'age'] = media

#X preditores - y classes 0 e 1
X = base.iloc[:,0:4].values
y = base.iloc[:,4].values

#Normalizando os dados
normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)

#Tamanho ideal da matriz
m = int((5*(len(base))**0.5)**0.5)//1
print(m)
n_p= (len(X[1]))
print(n_p)

#criação do mapa
som = MiniSom(x = m, y = m, input_len = n_p , sigma = 2.0, learning_rate = 0.5, random_seed=0)
som.random_weights_init(X) #peso randomicos
som.train_random(data = X, num_iteration=200) #base, epochs =1000

#pesos e valores
som._weights
som._activation_map
numeroDeNeuronioSelecionado = som.activation_response(X)

#Mapa com as cores - MID mean inter neuron distance
pcolor(som.distance_map().T) #T transposição

#As colorações amarelas indicam que os neuronios não são tao confiaveis, pois estao 
#distantes dos seus vizinhos mais proximos

pcolor(som.distance_map().T) #T transposição
colorbar()
#atribuindo os resultados no mapa
#w = som.winner(X)
markers = ['o','s']
color = ['g','r']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5,
         markers[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgecolor = color[y[i]],
         markeredgewidth = 2)
    
# Os reistros que escolheram a coloração amarela estão fugindo do padrão, precisamos dar a
# devida atenção para que não seja uma fraude
# Quadrado vermelho não concedeu credito
# circulo verde concedeu crédito

# Clientes que não se enquadram, podem ser possiveis fraudes
mapeamento = som.win_map(X)
# Lista de coordenadas
coordenadas = [(1, 3), (6, 3), (1, 12)]  # Adicione quantas coordenadas desejar

# Concatenar os resultados para as coordenadas especificadas
suspeitos = np.concatenate([mapeamento[coord] for coord in coordenadas], axis=0)
suspeitos = normalizador.inverse_transform(suspeitos)

classe = []
for i in range(len(base)):
    for j in range(len(suspeitos)):
        if base.iloc[i,0] == int(round(suspeitos[j,0])): #Mesmo codigo base e suspeitos
            classe.append(base.iloc[i,4])
classe = np.asarray(classe)

suspeitos_final = np.column_stack((suspeitos, classe))
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]