# Ethereum Price Prediction with LSTM

Este projeto utiliza redes neurais LSTM para prever os preços da criptomoeda Ethereum com base em dados históricos.

## Estrutura

- Modelo LSTM com camadas empilhadas
- Pré-processamento com MinMaxScaler
- Geração de gráficos (treino, teste, geral)
- Avaliação com métricas MSE, MAE, MAPE e RMSE
- Treinamento com callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Arquivo principal

- `lstm_model_otimizado_final.py`: contém todo o pipeline de dados, construção, treinamento e avaliação do modelo.

## Gráficos gerados

- `grafico_treino.png`
- `grafico_teste.png`
- `grafico_geral.png`

## Como rodar

```bash
python lstm_model_otimizado_final.py
```

## Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt
```
