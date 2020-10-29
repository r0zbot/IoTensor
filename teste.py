# %%
import tensorflow as tf
import pandas as pd

data = pd.read_csv('dataset/YearPredictionMSD.txt', sep=',', header=None)

print(data)
# %%
#print(data[:][0])
# %%
data_train = data.iloc[:463715]
data_test = data.iloc[463716:]
# %%
data_train.describe()   
# %%
# Criando a camada de aprendizado com o Keras
l0 = tf.keras.layers.Dense(units=12, input_shape=[90])
l1 = tf.keras.layers.Dense(units=24)
l2 = tf.keras.layers.Dense(units=1)
# Iniciando o modelo
model = tf.keras.Sequential( [ l0, l1, l2 ] )
# Compilando o modelo
model.compile(loss='mean_squared_error', # função que mede quão preciso é o modelo
              optimizer=tf.keras.optimizers.Adam(0.1))  # como o modelo se atualiza com base as amostras e na função de loss
# %%
# Treinando o modelo
import numpy as np

labels = np.array(data_train[0])
dados = np.array(data_train.iloc[:,1:].to_numpy())
print(labels.shape)
print(dados.shape)
history = model.fit(dados, labels, epochs=2, verbose=True)
print("Finished training the model")

model.predict(np.array([data_test.iloc[0, 1:]]))

# %%
# Avaliação do modelo
#_, acc = model.evaluate(data_test.iloc[:,1:].to_numpy(), verbose=2)
# Visualização
import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.title('Loss')
plt.plot(history.history['loss'], color='green', label='Modelo')
plt.subplots_adjust(top=2.0)
plt.legend()
plt.grid()
plt.show()

# %%
from tensorflow.keras import models, layers, optimizers
def configure_model_3():
    '''Modelo para inputs 2D, tem que mudar'''
    network = models.Sequential()
    network.add(layers.Conv1D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=([90])))
    network.add(layers.MaxPooling1D((2, 2)))
    network.add(layers.Flatten())
    network.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network

def run_model_3(model_3, stats_3, learn_x, learn_y, val_x, val_y):
    history = model_3.fit(learn_x.reshape((len(learn_x), 28, 28, 1)), learn_y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    _, acc = model_3.evaluate(val_x.reshape(len(val_x), 28, 28, 1), to_categorical(val_y), verbose=0)
    stats_3['history'].append(history)
    stats_3['accuracy'].append(acc)
    print(f'\tModelo 3: {acc:.3f}')

model_3 = configure_model_3()
# %%
