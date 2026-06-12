import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import random

arquivos = ["20150101.txt", "20150102.txt"]

variaveis = ["Duracao", "Servico", "Bytes_origem", "Bytes_destino", "Qtd", "Tx_msm_servico", "Tx_Serro",
             "Tx_Serro_servico", "Destino_qtd_host", "Destino_host_qtd_servico", "Destino_host_msm_tx_porta_origem",
             "Destino_host_tx_serro", "Destino_host_tx_serro_servico", "Flag", "Detec_IDS", "Detec_Malw",
             "Detec_Ashula", "Rotulo", "IP_Origem", "Porta_Origem", "IP_Destino", "Porta_Destino", "T_Comeco",
             "Protocolo"]

dados_leitura = [pd.read_csv(arq, sep="\t", header=None, names=variaveis) for arq in arquivos]
kyoto = pd.concat(dados_leitura, ignore_index=True)

kyoto = kyoto[kyoto['Rotulo'] != -2]

filtro = ["Rotulo", "Duracao", "Servico", "Bytes_origem", "Bytes_destino",
          "Qtd", "Destino_qtd_host", "Destino_host_qtd_servico",
          "Destino_host_tx_serro", "Flag", "Protocolo"]

kyotoFiltrada = kyoto[filtro].dropna().copy()

kyotoFiltrada['Rotulo'] = kyotoFiltrada['Rotulo'].replace({-1: 1, 1: 0})

y = kyotoFiltrada['Rotulo']
X = kyotoFiltrada.drop('Rotulo', axis=1)

X = pd.get_dummies(X, columns=['Servico', 'Flag', 'Protocolo'], drop_first=True)

X = X.astype(np.float32)
y = y.astype(np.float32)

params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.05,
    'num_leaves': 50,
    'max_depth': -1,
    'device_type': 'gpu',
    'gpu_use_dp': False,
    'verbose': -1
}

num_rodadas = 1000

acuracias_iteracao = []
tempos_iteracao = []
acuracias_cumulativas = []
tempos_cumulativos = []

for i in range(num_rodadas):
    seed_atual = random.randint(0, 2 ** 32 - 1)

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=seed_atual)

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)

    dados_treino = lgb.Dataset(X_treino_scaled, label=y_treino)
    dados_teste = lgb.Dataset(X_teste_scaled, label=y_teste, reference=dados_treino)

    inicio = time.time()
    modelo = lgb.train(
        params,
        dados_treino,
        num_boost_round=500,
        valid_sets=[dados_treino, dados_teste],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    fim = time.time()

    probabilidades = modelo.predict(X_teste_scaled, num_iteration=modelo.best_iteration)
    y_pred = np.where(probabilidades > 0.5, 1, 0)

    acuracia = accuracy_score(y_teste, y_pred)
    tempo_exec = fim - inicio

    acuracias_iteracao.append(acuracia)
    tempos_iteracao.append(tempo_exec)

    acuracias_cumulativas.append(np.mean(acuracias_iteracao))
    tempos_cumulativos.append(np.mean(tempos_iteracao))

resultados = pd.DataFrame({
    'ite': range(1, num_rodadas + 1),
    'Acuracia': acuracias_iteracao,
    'tExecS': tempos_iteracao,
    'MediaAcCumulativa': acuracias_cumulativas,
    'MediaTCumulativa': tempos_cumulativos
})

resultados.to_csv('resultados_monte_carlo_lightgbm.csv', index=False)