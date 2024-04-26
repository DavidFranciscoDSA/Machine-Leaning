# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 08:50:59 2024

@author: david
"""

#Importação das bibliotecas necessárias
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import Sequential
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import cv2

import datetime
def hora():
    hora_atual = datetime.datetime.now()
    hora_formatada = hora_atual.strftime("%H:%M:%S")
    print(hora_formatada)
    
#Etapas de 1 a 3 - dimensao da imagem 64 x 64 4096 pixels - 3 por ser padrao RGB
classificador = Sequential()
classificador.add(Conv2D(128, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(256, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(512, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))


classificador.add(Flatten())
    
#Camadas densas
classificador.add(Dense(units = 512, activation = 'relu'))
classificador.add(Dropout(0.20))
classificador.add(Dense(units = 256, activation = 'relu'))
classificador.add(Dropout(0.20))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.20))
classificador.add(Dense(units = 64, activation = 'relu'))
classificador.add(Dropout(0.20))

classificador.add(Dense(units = 1, activation = 'sigmoid')) #Sigmoid retorna probabilidade

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

#Normalização das imagens de 1 a 255 para 0 a 1
gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

#base de treinamento e teste
base_treinamento = gerador_treinamento.flow_from_directory('plantas/treino',
                                                            target_size = (64, 64),
                                                            batch_size = 5,
                                                            class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('plantas/teste',
                                               target_size = (64, 64),
                                               batch_size = 5,
                                               class_mode = 'binary')
#Treinamento
hora()

classificador.fit(base_treinamento, steps_per_epoch=829 // 5,
                  epochs=100, validation_data=base_teste,
                  validation_steps=262 // 5)

hora()

def busca_por_input():
    #buscando a imagem a ser testada
    diretorio = 'plantas/teste/'
    pasta = int(input('Digite [0] doentes e [1] saudáveis'))
    r = pasta
    while pasta == 0 or pasta == 1:
        if pasta == 1:
            r = 1
            pasta = 'saudaveis'
        else:
            r = 0
            pasta = 'doentes'
        img = input('nome do arquivo: ex: sau10035.jpg ')
        caminho = diretorio + pasta + '/' + img

        #caminho do arquivo
        imagem_teste = image.load_img(caminho,
                                      target_size = (64,64))
        #convertendo a imagem (64,64,3) - mesmo formado em que foi usado acima e normalizando 0 a 1
        imagem_teste = image.img_to_array(imagem_teste)
        imagem_teste /= 255
        #Colocando no formato que o tensorflow trabalha (1, 64 ,64,3) - 1 é o batch de quantidade de imagens
        imagem_teste = np.expand_dims(imagem_teste, axis = 0)
        #a linha abaixo mostra a previsao da imagem 0 a 1 - cachorro[0] gato[1]
        previsao = classificador.predict(imagem_teste)
        #acima de 50%
        print(previsao)
        print(caminho)

        imagem_teste = cv2.imread(caminho, cv2.IMREAD_COLOR)
        plt.imshow(imagem_teste)
        plt.show()
        if previsao > 0.5:
            print('A planta aparenta estar saudável.')
        else:
            print('A planta aparenta estar com pragas ou doenças.')
        base_treinamento.class_indices
        print()
        pasta = int(input('Digite [0] doentes e [1] saudáveis, ou outro número para sair: '))



hora()
Y_real = []
y_probabilidade = []
dteste_doente = 0
dteste_saude = 0
diretorio = 'plantas/teste/doentes/'
arquivo = 'doe100'
teste_arquivo = []
for i in range(131): #numero de imagens na pasta teste
    if i < 10:
        arquivo = 'doe100'
        arquivo = arquivo + '0' + str(i) + '.jpg'
        caminho = diretorio + arquivo
    elif i >= 10 and i < 100:
        arquivo = 'doe100'
        arquivo = arquivo +  str(i) + '.jpg'
        caminho = diretorio + arquivo
    else:
        arquivo = 'doe10'
        arquivo = arquivo +  str(i) + '.jpg'
        caminho = diretorio + arquivo
    imagem_teste = image.load_img(caminho,target_size = (64,64))
    #convertendo a imagem (64,64,3) - mesmo formado em que foi usado acima e normalizando 0 a 1
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste /= 255
    #Colocando no formato que o tensorflow trabalha (1, 64 ,64,3) - 1 é o batch de quantidade de imagens
    imagem_teste = np.expand_dims(imagem_teste, axis = 0)
    #a linha abaixo mostra a previsao da imagem 0 a 1 - doente[0] saudavel[1]
    previsao = classificador.predict(imagem_teste)
    #acima de 50%
    #print(previsao)
    if previsao > 0.5:
        #print('A planta aparenta estar saudável.')
        dteste_saude+=1
        #print(caminho)
        teste_arquivo.append(arquivo)
    else:
        #print('A planta aparenta estar com pragas ou doença.')
        dteste_doente+=1
        #print(caminho)
        #base_treinamento.class_indices
    Y_real.append(0)
    y_probabilidade.append(previsao)
print()
print("Predição de plantas doentes")
print(f'Doente: {dteste_doente}')
print(f'Saudavel: {dteste_saude}')


steste_doente = 0
steste_saude = 0
diretorio = 'plantas/teste/saudaveis/'
arquivo = 'sau100'
for i in range(131): #numero de imagens na pasta teste
    if i < 10:
        arquivo = 'sau100'
        arquivo = arquivo + '0' + str(i) + '.jpg'
        caminho = diretorio + arquivo
    elif i >= 10 and i < 100:
        arquivo = 'sau100'
        arquivo = arquivo +  str(i) + '.jpg'
        caminho = diretorio + arquivo
    else:
        arquivo = 'sau10'
        arquivo = arquivo +  str(i) + '.jpg'
        caminho = diretorio + arquivo
    imagem_teste = image.load_img(caminho,target_size = (64,64))
    #convertendo a imagem (64,64,3) - mesmo formado em que foi usado acima e normalizando 0 a 1
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste /= 255
    #Colocando no formato que o tensorflow trabalha (1, 64 ,64,3) - 1 é o batch de quantidade de imagens
    imagem_teste = np.expand_dims(imagem_teste, axis = 0)
    #a linha abaixo mostra a previsao da imagem 0 a 1 - doente[0] saudavel[1]
    previsao = classificador.predict(imagem_teste)
    #acima de 50%
    #print(previsao)
    if previsao > 0.5:
        #print('A planta aparenta estar saudável.')
        steste_saude+=1
        #print(caminho)
    else:
        #print('A planta aparenta estar com pragas ou doença.')
        steste_doente+=1
        teste_arquivo.append(arquivo)
        # print(caminho)
    #base_treinamento.class_indices
    Y_real.append(1)
    y_probabilidade.append(previsao)
    
print()
print("Predição de plantas saudáveis") 
print(f'Doente: {steste_doente}')
print(f'Saudavel: {steste_saude}')
hora()


print(len(Y_real))
# Extrair os valores numéricos
valores_numericos = [float(array[0][0]) for array in y_probabilidade]
y_prob = valores_numericos
# Imprimir os valores extraídos
print(len(y_prob))


#Calculos das métricas
sensibilidade_Recall = steste_saude / (steste_saude + dteste_saude) 
tx_FP = steste_doente / (steste_doente + dteste_doente) 
especificidade = dteste_doente / (dteste_doente + steste_doente) 
tx_FN = dteste_saude / (dteste_saude + steste_saude)
acuracia = (steste_saude + dteste_doente)/\
           (steste_saude + dteste_doente + 
            dteste_saude + steste_doente)
precisao = steste_saude / (steste_saude + steste_doente)
f1score = 2*sensibilidade_Recall*precisao / (sensibilidade_Recall + precisao)
Taxa_erro = (steste_doente + dteste_saude)/\
           (steste_saude + dteste_doente + 
            dteste_saude + steste_doente)
#Impressão do resultado
print(f'''           Acertos:                ERROS
Saudaveis:  \t{steste_saude}  \t  -          {steste_doente}
Doentes:     \t{dteste_doente} \t  -          {dteste_saude}

Verdadeiro positivo: {steste_saude}   \t     \t→ Modelo previu e acertou os dados como positivos.
Falso positivo:\t     {steste_doente}\t          \t→ Modelo previu como negativo, mas era positivo.
Verdadeiro negativo: {dteste_doente}\t \t    \t→ Modelo previu e acertou os dados como negativo.
Falso negativo:\t     {dteste_saude}          \t→ Modelo previu como positivo, mas era negativo.

''')
print('Métricas do modelo')
print(f'''
Acurácia: \t \t {acuracia:.4f}     \t→ Indica que o modelo acertou {(acuracia*100):.2f}% das previsões
Taxa De Erro: \t \t {Taxa_erro:.4f}     \t→ Indica que o modelo errou {(Taxa_erro*100):.2f}% das previsões
Sensibilidade (Recall):  {sensibilidade_Recall:.4f} \t→ O modelo identificou corretamente {(sensibilidade_Recall*100):.2f}% dos casos positivos.
Especificidade: \t {especificidade:.4f}     \t→ Indica que o modelo identificou corretamente {(especificidade*100):.2f}% dos casos negativos.        
Precisão: \t \t {precisao:.4f}  \t→ Indica que {(precisao*100):.2f}% das instâncias previstas como positivas eram realmente positivas.
F1-Score: \t\t {f1score:.4f}    \t→ Indicando um equilíbrio entre precisão e recall de {(f1score*100):.2f}%
''')    

# Previstos (y_prob)  reais (y_true)
tx_FP, sensibilidade_Recall, thresholds = roc_curve(Y_real, y_prob)

# Calcular a área sob a curva ROC (AUC-ROC)
roc_auc = auc(tx_FP, sensibilidade_Recall)
print(f'Área sobre a curva ROC: {roc_auc:.2f}%')
# Plotar a curva ROC
plt.figure(figsize=(5, 5))
plt.plot(tx_FP, sensibilidade_Recall, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()


#usando dados reais
hora()
Y_real = []
y_probabilidade = []
dteste_doente = 0
dteste_saude = 0
diretorio = 'Estacao/doentes/'
arquivo = 'doe'
teste_arquivo = []
for i in range(1,31): #numero de imagens na pasta teste
    if i < 31:
        arquivo = 'doe'
        arquivo = arquivo  + str(i) + '.jpg'
        caminho = diretorio + arquivo
    elif i >= 31 and i < 100:
        arquivo = 'doe100'
        arquivo = arquivo +  str(i) + '.jpg'
        caminho = diretorio + arquivo
    else:
        arquivo = 'doe10'
        arquivo = arquivo +  str(i) + '.jpg'
        caminho = diretorio + arquivo
    imagem_teste = image.load_img(caminho,target_size = (64,64))
    #convertendo a imagem (64,64,3) - mesmo formado em que foi usado acima e normalizando 0 a 1
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste /= 255
    #Colocando no formato que o tensorflow trabalha (1, 64 ,64,3) - 1 é o batch de quantidade de imagens
    imagem_teste = np.expand_dims(imagem_teste, axis = 0)
    #a linha abaixo mostra a previsao da imagem 0 a 1 - doente[0] saudavel[1]
    previsao = classificador.predict(imagem_teste)
    #acima de 50%
    #print(previsao)
    if previsao > 0.5:
        #print('A planta aparenta estar saudável.')
        dteste_saude+=1
        #print(caminho)
        teste_arquivo.append(arquivo)
    else:
        #print('A planta aparenta estar com pragas ou doença.')
        dteste_doente+=1
        #print(caminho)
        #base_treinamento.class_indices
    Y_real.append(0)
    y_probabilidade.append(previsao)
print()
print("Predição de plantas doentes")
print(f'Doente: {dteste_doente}')
print(f'Saudavel: {dteste_saude}')

steste_doente = 0
steste_saude = 0
diretorio = 'Estacao/saudaveis/'
arquivo = 'sau'
for i in range(1,21): #numero de imagens na pasta teste
    if i < 21:
        arquivo = 'sau'
        arquivo = arquivo  + str(i) + '.jpg'
        caminho = diretorio + arquivo
    elif i >= 21 and i < 100:
        arquivo = 'sau100'
        arquivo = arquivo +  str(i) + '.jpg'
        caminho = diretorio + arquivo
    else:
        arquivo = 'sau10'
        arquivo = arquivo +  str(i) + '.jpg'
        caminho = diretorio + arquivo
    imagem_teste = image.load_img(caminho,target_size = (64,64))
    #convertendo a imagem (64,64,3) - mesmo formado em que foi usado acima e normalizando 0 a 1
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste /= 255
    #Colocando no formato que o tensorflow trabalha (1, 64 ,64,3) - 1 é o batch de quantidade de imagens
    imagem_teste = np.expand_dims(imagem_teste, axis = 0)
    #a linha abaixo mostra a previsao da imagem 0 a 1 - doente[0] saudavel[1]
    previsao = classificador.predict(imagem_teste)
    #acima de 50%
    #print(previsao)
    if previsao > 0.5:
        #print('A planta aparenta estar saudável.')
        steste_saude+=1
        #print(caminho)
    else:
        #print('A planta aparenta estar com pragas ou doença.')
        steste_doente+=1
        teste_arquivo.append(arquivo)
        # print(caminho)
    #base_treinamento.class_indices
    Y_real.append(1)
    y_probabilidade.append(previsao)
    
print()
print("Predição de plantas saudáveis") 
print(f'Doente: {steste_doente}')
print(f'Saudavel: {steste_saude}')
hora()