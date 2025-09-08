import os

SEED = 18
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import random
import tensorflow as tf
import numpy as np

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, SpatialDropout1D, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import metrics, backend as K
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt


# Leitura e preparação dos dados
df = pd.read_csv('Ethereum.csv').dropna()
df['log_close'] = np.log(df['close'])
df['return']    = df['log_close'].diff()
df = df.dropna()
x = df[['open', 'high', 'low', 'volume']]
y = df['return']


split_idx = int(len(x) * 0.8)
x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

#num_steps = int(len(y_test) / 2)
num_steps = 60

normalizador_entrada = MinMaxScaler()
normalizador_saida = MinMaxScaler()

x_train_norm = normalizador_entrada.fit_transform(x_train)
y_train_norm = normalizador_saida.fit_transform(np.array(y_train).reshape(-1, 1))

lista_previsores, lista_preco_real = [], []
for i in range(num_steps, len(x_train_norm)):
    lista_previsores.append(x_train_norm[i - num_steps:i, :])
    lista_preco_real.append(y_train_norm[i])

lista_previsores = np.array(lista_previsores)
lista_preco_real = np.array(lista_preco_real)

num_features = x_train.shape[1]
lista_previsores = np.reshape(lista_previsores, (lista_previsores.shape[0], lista_previsores.shape[1], num_features))

split = int(len(lista_previsores) * 0.9)
x_tr, y_tr = lista_previsores[:split], lista_preco_real[:split]
x_val, y_val = lista_previsores[split:], lista_preco_real[split:]

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def build_cnn1d_model(filters_list, kernel_sizes, activation_conv, activation_dense, activation_out,
                      dropout_rate, learning_rate, optimizer_class, input_shape):
    model = Sequential()

    for i, (filters, kernel) in enumerate(zip(filters_list, kernel_sizes)):
        if i == 0:
            model.add(Conv1D(filters=filters,
                             kernel_size=kernel,
                             use_bias=False,
                             padding='causal',
                             input_shape=input_shape))
        else:
            model.add(Conv1D(filters=filters,
                             kernel_size=kernel,
                             use_bias=False,
                             padding='causal'))
        model.add(BatchNormalization())
        model.add(Activation(activation_conv))
        model.add(SpatialDropout1D(0.2))
        if i < len(filters_list) - 1:
            model.add(MaxPooling1D(pool_size=2))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation=activation_dense))
    model.add(Dense(1, activation=activation_out))

    optimizer = optimizer_class(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=Huber(),
                  metrics=[metrics.MeanAbsoluteError(),
                           metrics.MeanAbsolutePercentageError(),
                           rmse])
    return model

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

regressor = build_cnn1d_model(
    filters_list=[128,256,512],
    kernel_sizes=[5,5,3],
    activation_conv='swish',
    activation_dense='swish',
    activation_out='linear',
    dropout_rate=0.2,
    learning_rate=0.0005,
    optimizer_class=Adam,
    input_shape=(lista_previsores.shape[1], lista_previsores.shape[2])
)
regressor.fit(x_tr, y_tr,
              epochs=150,
              batch_size=16,
              validation_data=(x_val, y_val),
              shuffle=False,
              callbacks=[early_stop, reduce_lr],
              verbose=1)

# Preparação dos dados de teste
x_test_norm = normalizador_entrada.transform(x_test)
y_test_norm = normalizador_saida.transform(np.array(y_test).reshape(-1, 1))

lista_previsores_test, lista_preco_real_test = [], []
for i in range(num_steps, len(x_test_norm)):
    lista_previsores_test.append(x_test_norm[i - num_steps:i, :])
    lista_preco_real_test.append(y_test_norm[i])

lista_previsores_test = np.array(lista_previsores_test)
lista_preco_real_test = np.array(lista_preco_real_test)
lista_previsores_test = np.reshape(
    lista_previsores_test,
    (lista_previsores_test.shape[0], lista_previsores_test.shape[1], num_features)
)

# Avaliação interna em escala normalizada
eval = regressor.evaluate(lista_previsores_test, lista_preco_real_test, batch_size=32)

# Gera previsões normalizadas e inverte para reais
pred_teste_norm = regressor.predict(lista_previsores_test)
pred_teste_real = normalizador_saida.inverse_transform(pred_teste_norm)
y_test_real = normalizador_saida.inverse_transform(lista_preco_real_test)

# Reconstrução do preço a partir do retorno previsto
# Extrai o log_close correspondente ao conjunto de teste (já sem o first-diff)
log_close_test = df['log_close'].iloc[split_idx:].reset_index(drop=True)

# Para cada retorno previsto, pega o log_close do dia anterior:
prev_log_close = log_close_test.shift(1).iloc[num_steps:].values

# pred_teste_real é shape (N,1), vamos achatar pra (N,)
ret_pred = pred_teste_real.flatten()

# calcula o log_close previsto e converte pra preço
pred_log_close = prev_log_close + ret_pred
pred_price = np.exp(pred_log_close)

# Preço “real” de fechamento para comparação:
actual_logc = log_close_test.iloc[num_steps:].values
actual_price = np.exp(actual_logc)

# MAPE sobre o preço reconstruído
mape_price = np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100
print(f"MAPE preço (após reconstrução): {mape_price:.2f}%")

# Gráficos e previsões (mantidos iguais para treino)
pred_treino = regressor.predict(lista_previsores)
pred_treino = normalizador_saida.inverse_transform(pred_treino)
lista_preco_real = normalizador_saida.inverse_transform(lista_preco_real)

log_close_train = df['log_close'].iloc[:split_idx].reset_index(drop=True)
prev_log_close_train = log_close_train.shift(1).iloc[num_steps:].values
ret_pred_train = pred_treino.flatten()
pred_log_close_train = prev_log_close_train + ret_pred_train
pred_price_treino = np.exp(pred_log_close_train)
actual_logc_train = log_close_train.iloc[num_steps:].values
actual_price_treino = np.exp(actual_logc_train)

plt.figure(figsize=(10, 6))
plt.plot(pred_price_treino,     label='Predito (treino)')
plt.plot(actual_price_treino,   label='Real    (treino)')
plt.title('Previsão de Preço no Conjunto de Treino')
plt.xlabel('Eixo X: Amostra de teste')
plt.ylabel('Eixo Y: Preço (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_treino_cnn1d_preco.png', dpi=100)
plt.close()

# Gráfico do preço reconstruído no teste
plt.figure(figsize=(10, 6))
plt.plot(pred_price,  label='Predito (teste)')
plt.plot(actual_price, label='Real    (teste)')
plt.title('Previsão de Preço no Conjunto de Teste')
plt.xlabel('Eixo X: Amostra de teste')
plt.ylabel('Eixo Y: Preço (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_teste_cnn1d_preco.png', dpi=100)
plt.close()

# Previsão Geral (Treino + Teste) em preço
plt.figure(figsize=(12, 6))
plt.plot(
    np.concatenate((pred_price_treino, pred_price)),
    label='Predito (Geral)'
)
plt.plot(
    np.concatenate((actual_price_treino, actual_price)),
    label='Real    (Geral)'
)
plt.title('Previsão Geral de Preço')
plt.xlabel('Eixo X: Amostra de teste')
plt.ylabel('Eixo Y: Preço (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_geral_cnn1d_preco.png', dpi=100)
plt.close()


# Final Evaluation incluindo mape_real
print("\nFinal Evaluation:")
print("metrics = ['RMSE','MAE','MAPE']")
final_metrics = {
    'root_mean_squared_error': float(np.sqrt(eval[0])),
    'mean_absolute_error': eval[1],
    'mape_price': mape_price
}
print(final_metrics)

