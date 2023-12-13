import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import glob
import os
import random
import keras
from sklearn.metrics import classification_report
import tensorflow as tf
from keras import models
from sklearn.model_selection import train_test_split


num_classes = 4

def sequential_mod(tupla_imagens: list):
    ''' Remapeia valor '''
    seq_array = []
    for i in tupla_imagens:
        # num de codigos para falhas: 
        # 1\normal, 2\pressão, 3\compressao, 4\combustivel
        if 'normal' in i[0]           : val = 0 # precisa começar com 0 pois eh zero-indexed
        elif 'red_pressao' in i[0]    : val = 1
        elif 'red_razcomp' in i[0]    : val = 2
        elif 'red_comb' in i[0]       : val = 3
        else                          : raise Exception("Categoria n encontrada")
        # print(f"-imagem: {i[0]}    valor: {val}")
        seq_array.append(val)
    return seq_array

def load_data(path) -> list:
    data = []
    files = glob.glob(path)
    for my_file in files:
        # print(f"myfile: {my_file}")
        image = Image.open(my_file).convert('RGB')
        image_data = np.array(image)
        # Add nome da imagem: './Plots_STFT/imagens_treino\\normal_1.png', pega so normal_1
        nome_imagem = my_file.split("\\")[-1].split('.png')[0]
        dados_imagem = (nome_imagem, image_data/255)
        data.append(dados_imagem)
    return data

# train_X = load_data(r'.\Plots_STFT\imagens_treino\*.png') # 
# test_X = load_data(r'.\Plots_STFT\imagens_teste\*.png') # 
dados_tratados_0_X = load_data(r'C:\Users\gabri\OneDrive\Faculdade\TCC\Codigo e dados\Plots_STFT\normal\*.png') 
dados_tratados_1_X = load_data(r'C:\Users\gabri\OneDrive\Faculdade\TCC\Codigo e dados\Plots_STFT\red_pressao\*.png') 
dados_tratados_2_X = load_data(r'C:\Users\gabri\OneDrive\Faculdade\TCC\Codigo e dados\Plots_STFT\red_razcomp\*.png') 
dados_tratados_3_X = load_data(r'C:\Users\gabri\OneDrive\Faculdade\TCC\Codigo e dados\Plots_STFT\red_comb\*.png') 

dados_tratados_0_y = np.array(sequential_mod(dados_tratados_0_X))
dados_tratados_1_y = np.array(sequential_mod(dados_tratados_1_X))
dados_tratados_2_y = np.array(sequential_mod(dados_tratados_2_X))
dados_tratados_3_y = np.array(sequential_mod(dados_tratados_3_X))

# Junta tudo agora
dados_tratados_concatenado_X = dados_tratados_0_X + dados_tratados_1_X + dados_tratados_2_X + dados_tratados_3_X
dados_tratados_concatenado_X = np.array([x[1] for x in dados_tratados_concatenado_X])
# dados_tratados_concatenado_X = np.concatenate((dados_tratados_0_X, dados_tratados_1_X, dados_tratados_2_X, dados_tratados_3_X))
dados_tratados_concatenado_y = np.concatenate((dados_tratados_0_y, dados_tratados_1_y, dados_tratados_2_y, dados_tratados_3_y))

X_train, X_test, y_train, y_test = train_test_split(
                                                    dados_tratados_concatenado_X, 
                                                    dados_tratados_concatenado_y, 
                                                    test_size       = 0.25,
                                                    random_state    = 42)

# Change the labels from categorical to one-hot encoding  
train_Y_one_hot = tf.keras.utils.to_categorical(y_train)
test_Y_one_hot = tf.keras.utils.to_categorical(y_test)

#load model
CNN = keras.models.load_model(r'C:\Users\gabri\OneDrive\Faculdade\TCC\Codigo e dados\Execucoes\CNN_STFT_tuner_24_10_2023.h5')

######## evaluate the model ########
_, train_acc = CNN.evaluate(X_train,train_Y_one_hot, verbose=0)
_, test_acc  = CNN.evaluate(X_test, test_Y_one_hot, verbose=0)
print('Train: %.4f, Test: %.4f' % (train_acc, test_acc))

predicted_classes = CNN.predict([X_test])
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

predicted_classes.shape, y_test.shape
print('Classification report:')
print(classification_report(y_test,predicted_classes,target_names=['1','2','3','4'],digits=4))

# Feature visualization
img = Image.open(r"C:\Users\gabri\OneDrive\Faculdade\TCC\Codigo e dados\Plots_STFT\red_pressao\red_pressao_15.png").convert('RGB')
img_tensor = np.array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor = img_tensor / 255

# Outputs of the layers, which include conv2D and max pooling layers
layer_outputs = [layer.output for layer in CNN.layers[:9]]
activation_model = models.Model(inputs = CNN.input, outputs = layer_outputs)
activations = activation_model.predict(img_tensor)
     
layer_names = []
for layer in CNN.layers[:12]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis('off')
    plt.imshow(display_grid, aspect='auto', cmap='jet')
plt.show()
