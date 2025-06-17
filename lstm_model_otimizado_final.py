import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop
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

len_dataset = len(y_test) + len(y_train)
num_steps = int(len(y_test) / 2)
#num_steps = 60


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

# Função de métrica customizada
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Função para construção do modelo LSTM
def build_lstm_model(units_list, activation_lstm, activation_out, learning_rate, optimizer_class):
    model = Sequential()
    for i, units in enumerate(units_list):
        return_seq = i < len(units_list) - 1
        if i == 0:
            model.add(LSTM(units,
                           return_sequences=return_seq,
                           activation=activation_lstm,
                           input_shape=(lista_previsores.shape[1], lista_previsores.shape[2])))
        else:
            model.add(LSTM(units,
                           return_sequences=return_seq,
                           activation=activation_lstm))
        if return_seq:
            model.add(Dropout(0.2))
    model.add(Dense(1, activation=activation_out))
    optimizer = optimizer_class(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=[metrics.MeanAbsoluteError(), metrics.MeanAbsolutePercentageError(), rmse])
    return model

# Callbacks
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.5, verbose=1)

# Criação do modelo
regressor = build_lstm_model(
    units_list=[256, 128, 96,64, 32],
    activation_lstm='tanh',
    activation_out='linear',
    learning_rate=0.001,
    optimizer_class=Adam,
)

# Treinamento
regressor.fit(lista_previsores, lista_preco_real,
              epochs=100,
              batch_size=8,
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
lista_previsores_test = np.reshape(lista_previsores_test, (lista_previsores_test.shape[0], lista_previsores_test.shape[1], num_features))

# Avaliação interna em escala normalizada
eval = regressor.evaluate(lista_previsores_test, lista_preco_real_test, batch_size=32)

# gera previsões normalizadas e inverte para reais
pred_teste_norm = regressor.predict(lista_previsores_test)
pred_teste_real = normalizador_saida.inverse_transform(pred_teste_norm)
y_test_real     = normalizador_saida.inverse_transform(lista_preco_real_test)

# calcula o MAPE real em reais
mape_real = np.mean(
    np.abs((y_test_real - pred_teste_real) / y_test_real)
) * 100

# Gráficos e previsões (mantidos iguais)
pred_treino = regressor.predict(lista_previsores)
pred_treino = normalizador_saida.inverse_transform(pred_treino)
lista_preco_real = normalizador_saida.inverse_transform(lista_preco_real)

plt.figure(figsize=(10, 6))
plt.plot(pred_treino.flatten(), label='Predito (treino)')
plt.plot(lista_preco_real.flatten(), label='Real (treino)')
plt.title('Previsão no Conjunto de Treino')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_treino.png', dpi=100)
plt.close()

pred_teste = pred_teste_real  # já em escala real
plt.figure(figsize=(10, 6))
plt.plot(pred_teste.flatten(), label='Predito (teste)')
plt.plot(y_test_real.flatten(), label='Real (teste)')
plt.title('Previsão no Conjunto de Teste')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_teste.png', dpi=100)
plt.close()

todas_predicoes = np.concatenate((pred_treino.flatten(), pred_teste.flatten()))
todos_precos_reais = np.concatenate((lista_preco_real.flatten(), y_test_real.flatten()))

plt.figure(figsize=(12, 6))
plt.plot(todas_predicoes, label='Predito (Geral)')
plt.plot(todos_precos_reais, label='Real (Geral)')
plt.title('Previsão Geral (Treino + Teste)')
plt.legend()
plt.tight_layout()
plt.savefig('grafico_geral.png', dpi=100)
plt.close()

# Impressão de comparações
print("\nTreino - Comparação Predito vs Real:")
for i in range(len(lista_preco_real)):
    print(f"pred: {pred_treino[i][0]:.2f}  real: {lista_preco_real[i][0]:.2f}")

print("\nTeste - Comparação Predito vs Real:")
for i in range(len(y_test_real)):
    print(f"pred: {pred_teste[i][0]:.2f}  real: {y_test_real[i][0]:.2f}")

# Final Evaluation incluindo MAPE real
print("\nFinal Evaluation:")
print("metrics = ['MSE','MAE','internal MAPE','RMSE','real MAPE']")
final_metrics = {
    'mean_squared_error': eval[0],
    'mean_absolute_error': eval[1],
    'internal_mape': eval[2],                   # MAPE sobre y_norm
    'root_mean_squared_error': float(np.sqrt(eval[0])),
    'mape_real': mape_real                      # MAPE em reais
}
print(final_metrics)

