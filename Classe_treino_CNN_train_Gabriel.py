# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import keras_tuner
from PIL import Image
import glob
import keras
import numpy as np
import sys
import tensorflow as tf
np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split
import datetime

# lr            = 0.001
# batch_size    = 16
# epochs        = 1000
# num_classes   = 4

# RUN = 'STFT'
# # RUN = 'CWTS'
# WITH_NOISE = True
# NOISE_VALUE = 75 #%

# hoje = datetime.datetime.today().strftime(r'%d_%m_%Y__%H_%M_%S')

########## UTILS
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
def sequential_mod(tupla_imagens: list) -> list:
    ''' Remapeia valor e monta lista sequencial com tags correspondentes a cada item lido. Ex: se for normal, add numero 0 à lista de dados
    input: tupla no formato (<nome_arquivo>, <dados_da_imagem>)'''
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

class RunCNNModel_TCC:
    def __init__(
            self, 
            TRANSFORMADA,
            WITH_NOISE,
            NOISE_VALUE   : int = 0,
            lr            : float = 0.001,
            batch_size    : float = 16,
            epochs        : float = 1000,
            num_classes   : float = 4,
) -> None:
        self.TRANSFORMADA   = TRANSFORMADA
        self.WITH_NOISE     = WITH_NOISE
        self.NOISE_VALUE    = NOISE_VALUE
        if self.NOISE_VALUE == 0:
            self.WITH_NOISE = False
        self.lr             = lr
        self.batch_size     = batch_size
        self.epochs         = epochs
        self.num_classes    = num_classes

        self.hoje           = datetime.datetime.today().strftime(r'%d_%m_%Y__%H_%M_%S')

    ################# MAIN FUNCTIONS
    def import_and_split_data(self,):
        
        dados_tratados_0_X = load_data(rF'C:\Users\gabri\OneDrive\Faculdade\TCC\Codigo e dados\NEW_Plots_{self.TRANSFORMADA}_w_noise\noise_{self.NOISE_VALUE}%\normal\*.png') 
        dados_tratados_1_X = load_data(rF'C:\Users\gabri\OneDrive\Faculdade\TCC\Codigo e dados\NEW_Plots_{self.TRANSFORMADA}_w_noise\noise_{self.NOISE_VALUE}%\red_pressao\*.png') 
        dados_tratados_2_X = load_data(rF'C:\Users\gabri\OneDrive\Faculdade\TCC\Codigo e dados\NEW_Plots_{self.TRANSFORMADA}_w_noise\noise_{self.NOISE_VALUE}%\red_razcomp\*.png') 
        dados_tratados_3_X = load_data(rF'C:\Users\gabri\OneDrive\Faculdade\TCC\Codigo e dados\NEW_Plots_{self.TRANSFORMADA}_w_noise\noise_{self.NOISE_VALUE}%\red_comb\*.png') 

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
        y_train_one_hot = tf.keras.utils.to_categorical(y_train)
        y_test_one_hot  = tf.keras.utils.to_categorical(y_test)

        return X_train, X_test, y_train_one_hot, y_test_one_hot, y_train, y_test


    def set_and_run_keras_tuner(self, 
            X_train, X_test, 
            y_train_one_hot, y_test_one_hot):
        
        def build_CNN_model(hp):
            # print("Criando rede neural...")
            CNN = Sequential()
            # ------ camada 1
            CNN.add(Conv2D(filters=16, kernel_size=(5, 5), strides = (2,2),
                        kernel_regularizer=tf.keras.regularizers.L2(hp.Float("regularizers_L2", min_value=1e-4, max_value=1e-1, sampling="log")),
                        activation=tf.keras.layers.LeakyReLU(alpha=0.3),input_shape=(128,128,3),padding='same'))
            CNN.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
            CNN.add(Dropout(rate=hp.Float("dropout_rate_1", min_value=0.015, max_value=0.15, sampling="log"))) # tentar diminuir, eh como se fosse ruido na rede
            # ------ camada 2
            CNN.add(Conv2D(filters=32, kernel_size=(5, 5), strides = (2,2),
                        kernel_regularizer=tf.keras.regularizers.L2(hp.Float("regularizers_L2", min_value=1e-4, max_value=1e-1, sampling="log")),
                        activation=tf.keras.layers.LeakyReLU(alpha=0.3),padding='same'))
            CNN.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
            CNN.add(Dropout(rate=hp.Float("dropout_rate_2", min_value=0.015, max_value=0.15, sampling="log")))
            # ------ camada 3
            CNN.add(Conv2D(filters=64, kernel_size=(5, 5), strides = (2,2),
                        kernel_regularizer=tf.keras.regularizers.L2(hp.Float("regularizers_L2", min_value=1e-4, max_value=1e-1, sampling="log")),
                        activation=tf.keras.layers.LeakyReLU(alpha=0.3),padding='same'))
            CNN.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
            CNN.add(Dropout(rate=hp.Float("dropout_rate_3", min_value=0.015, max_value=0.15, sampling="log")))
            # ------ camada final
            CNN.add(Flatten())
            CNN.add(Dense(
                units = hp.Int("Dense_units", min_value=16, max_value=512, step=16), #128,
                activity_regularizer = tf.keras.regularizers.L2(hp.Float("regularizers_L2", min_value=1e-4, max_value=1e-1, sampling="log")),
                activation = tf.keras.layers.LeakyReLU(alpha=0.3))
                )
            CNN.add(Dense(self.num_classes, activation='softmax'))
            CNN.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=self.lr),metrics=['accuracy'])
            return CNN
        
        tuner = keras_tuner.BayesianOptimization(
            hypermodel              = build_CNN_model,
            objective               = "val_accuracy",
            max_trials              = 10,
            executions_per_trial    = 2,
            overwrite               = True,
            directory               = "hyperpar_tunning",
            project_name            = "tentativa_1",
            # num_initial_points=None,
            # alpha=0.0001,
            # beta=2.6,
            # seed=None,
            # hyperparameters=None,
            # tune_new_entries=True,
            # allow_new_entries=True,
            # max_retries_per_trial=0,
            # max_consecutive_failed_trials=3,
        )
        es = EarlyStopping(
            monitor                = 'val_loss',
            mode                   = 'auto',
            min_delta              = 0.1,
            verbose                = 2,
            patience               = 10,
            restore_best_weights   = True
            )

        tuner.search_space_summary()

        tuner.search(
            X_train, 
            y_train_one_hot, 
            batch_size = self.batch_size, 
            epochs = self.epochs, 
            verbose = 1, 
            validation_data = (X_test, y_test_one_hot), 
            callbacks = [es])
        
        # Get the top 2 models.
        models = tuner.get_best_models(num_models=2)
        best_model = models[0]
        # Build the model.
        # Needed for `Sequential` without specified `input_shape`.
        best_model.build(input_shape=(None, 28, 28))

        best_model.summary()
        # if self.WITH_NOISE:
        #     best_model.save(f'./Execucoes/CNN_{self.TRANSFORMADA}_tuner_{self.hoje}_w_noise_{self.NOISE_VALUE}.h5')
        # else:
        #     best_model.save(f'./Execucoes/CNN_{self.TRANSFORMADA}_tuner_{self.hoje}.h5')   

        best_model.save(f'./Execucoes/CNN_{self.TRANSFORMADA}_tuner_{self.hoje}_w_noise_{self.NOISE_VALUE}.h5')   

        return None
