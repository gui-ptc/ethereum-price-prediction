import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics, backend as K
import matplotlib.pyplot as plt

# Leitura e preparação dos dados
df = pd.read_csv('Ethereum.csv')
df = df.dropna()
x = df[['open', 'high', 'low', 'volume']]
y = df['close']

split_idx = int(len(x) * 0.8)
x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

num_steps = int(len(y_test) / 2)

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

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Construção aprimorada do modelo CNN1D
def build_cnn1d_model(filters_list, kernel_sizes, activation_conv, activation_dense, activation_out,
                      dropout_rate, learning_rate, optimizer_class, input_shape):
    model = Sequential()

    # Camadas convolucionais + BN + MaxPool + Dropout
    for i, (filters, kernel) in enumerate(zip(filters_list, kernel_sizes)):
        if i == 0:
            model.add(Conv1D(filters=filters, kernel_size=kernel, activation=activation_conv, input_shape=input_shape))
        else:
            model.add(Conv1D(filters=filters, kernel_size=kernel, activation=activation_conv))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

    # Dropout apenas após o flatten, como na versão anterior
    model.add(Flatten())
    model.add(Dropout(dropout_rate))

    # Camadas densas finais
    model.add(Dense(64, activation=activation_dense))
    model.add(Dense(1, activation=activation_out))

    # Compilação
    optimizer = optimizer_class(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=[metrics.MeanAbsoluteError(), metrics.MeanAbsolutePercentageError(), rmse])
    return model

early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.5, verbose=1)

regressor = build_cnn1d_model(
    filters_list=[32, 64, 128],
    kernel_sizes=[5, 3, 3],
    activation_conv='relu',
    activation_dense='relu',
    activation_out='linear',
    dropout_rate=0.5,  # POS-Flatten agora
    learning_rate=0.001,
    optimizer_class=Adam,
    input_shape=(lista_previsores.shape[1], lista_previsores.shape[2])
)
regressor.fit(lista_previsores, lista_preco_real,
              epochs=150,
              batch_size=32,
              callbacks=[early_stop, reduce_lr],
              verbose=1)

x_test_norm = normalizador_entrada.transform(x_test)
y_test_norm = normalizador_saida.transform(np.array(y_test).reshape(-1, 1))

lista_previsores_test, lista_preco_real_test = [], []
for i in range(num_steps, len(x_test_norm)):
    lista_previsores_test.append(x_test_norm[i - num_steps:i, :])
    lista_preco_real_test.append(y_test_norm[i])

lista_previsores_test = np.array(lista_previsores_test)
lista_preco_real_test = np.array(lista_preco_real_test)
lista_previsores_test = np.reshape(lista_previsores_test, (lista_previsores_test.shape[0], lista_previsores_test.shape[1], num_features))

eval = regressor.evaluate(lista_previsores_test, lista_preco_real_test, batch_size=32)

pred_treino = regressor.predict(lista_previsores)
pred_treino = normalizador_saida.inverse_transform(pred_treino)
lista_preco_real = normalizador_saida.inverse_transform(lista_preco_real)

plt.figure(figsize=(10, 6))
plt.plot(pred_treino.flatten(), label='Predito (treino)')
plt.plot(lista_preco_real.flatten(), label='Real (treino)')
plt.title('Previsão no Conjunto de Treino')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_treino_cnn1d.png', dpi=100)
plt.close()

pred_teste = regressor.predict(lista_previsores_test)
pred_teste = normalizador_saida.inverse_transform(pred_teste)
lista_preco_real_test = normalizador_saida.inverse_transform(lista_preco_real_test)

plt.figure(figsize=(10, 6))
plt.plot(pred_teste.flatten(), label='Predito (teste)')
plt.plot(lista_preco_real_test.flatten(), label='Real (teste)')
plt.title('Previsão no Conjunto de Teste')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_teste_cnn1d.png', dpi=100)
plt.close()

todas_predicoes = np.concatenate((pred_treino.flatten(), pred_teste.flatten()))
todos_precos_reais = np.concatenate((lista_preco_real.flatten(), lista_preco_real_test.flatten()))

plt.figure(figsize=(12, 6))
plt.plot(todas_predicoes, label='Predito (Geral)')
plt.plot(todos_precos_reais, label='Real (Geral)')
plt.title('Previsão Geral (Treino + Teste)')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_geral_cnn1d.png', dpi=100)
plt.close()

print("\nFinal Evaluation:")
print("metrics = ['mean_squared_error(MSE)', 'mean_absolute_error(MAE)', 'mean_absolute_percentage_error(MAPE)', 'root_mean_squared_error(RMSE)']")
final_metrics = {
    'mean_squared_error': eval[0],
    'mean_absolute_error': eval[1],
    'mean_absolute_percentage_error': eval[2],
    'root_mean_squared_error': float(np.sqrt(eval[0]))
}
print(final_metrics)
