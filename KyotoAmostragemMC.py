import pandas as pd
import glob
import random
import lightgbm as lgb
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

variaveis = ["Duracao", "Servico", "Bytes_origem", "Bytes_destino", "Qtd", "Tx_msm_servico", "Tx_Serro",
             "Tx_Serro_servico", "Destino_qtd_host", "Destino_host_qtd_servico", "Destino_host_msm_tx_porta_origem",
             "Destino_host_tx_serro", "Destino_host_tx_serro_servico", "Flag", "Detec_IDS", "Detec_Malw",
             "Detec_Ashula", "Rotulo", "IP_Origem", "Porta_Origem", "IP_Destino", "Porta_Destino", "T_Comeco",
             "Protocolo"]

ano = "2012"
mes = "07"

caminho = f"dias/{ano}{mes}*.txt"
arquivos = glob.glob(caminho)

dadosCompletos = []
for arq in arquivos:
    dia = pd.read_csv(arq, sep="\t", header=None, names=variaveis)
    dadosCompletos.append(dia)

n = 1000
resAC = []
tempos = 0.0
resultados = []

for i in range(n):
    leitura = []

    for dia in dadosCompletos:
        qtdAmostras = random.randint(10000, 15000)
        qtdAmostras = min(qtdAmostras, len(dia))
        amostra = dia.sample(n=qtdAmostras, random_state=(123 + i))
        leitura.append(amostra)

    kyoto = pd.concat(leitura, ignore_index=True)
    kyoto = kyoto[kyoto['Rotulo'] != -2]

    filtro = ["Rotulo", "Duracao", "Servico", "Bytes_origem", "Bytes_destino",
              "Qtd", "Destino_qtd_host", "Destino_host_qtd_servico",
              "Destino_host_tx_serro", "Flag", "Protocolo"]

    kyotoFiltrada = kyoto[filtro].dropna().copy()
    kyotoFiltrada['Rotulo'] = kyotoFiltrada['Rotulo'].replace({-1: 1, 1: 0})

    y = kyotoFiltrada['Rotulo']
    X = kyotoFiltrada.drop('Rotulo', axis=1)

    # Transforma em flags de True ou False (0 1)
    X = pd.get_dummies(X, columns=['Servico', 'Flag', 'Protocolo'], drop_first=True)

    # Otimizações para GPU
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=(895769 + i))

    # Funciona como uma espécie de ajuste de importância
    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)

    dados_treino = lgb.Dataset(X_treino_scaled, label=y_treino)
    dados_teste = lgb.Dataset(X_teste_scaled, label=y_teste, reference=dados_treino)

    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'learning_rate': 0.1,
        'num_leaves': 50,
        'max_depth': -1,
        'device_type': 'gpu',
        'gpu_use_dp': False,
        'seed': i,
        'verbose': -1
    }

    inicio = time.time()
    modelo = lgb.train(
        params,
        dados_treino,
        num_boost_round=500,
        valid_sets=[dados_treino, dados_teste],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    fim = time.time()

    tempo_treino = fim - inicio
    tempos += tempo_treino
    tempoC = tempos / (i + 1)

    probabilidades = modelo.predict(X_teste_scaled, num_iteration=modelo.best_iteration)
    y_pred = np.where(probabilidades > 0.5, 1, 0)

    acuracia = accuracy_score(y_teste, y_pred)
    resAC.append(acuracia)

    acuracia_media_cumulativa = np.mean(resAC)

    resultados.append({
        "ite": i + 1,
        "Acuracia": acuracia,
        "tExecS": tempo_treino,
        "MediaAcCumulativa": acuracia_media_cumulativa,
        "MediaTCumulativa": tempoC
    })

df_historico = pd.DataFrame(resultados)
df_historico.to_csv("monteCarloA.csv", index=False)