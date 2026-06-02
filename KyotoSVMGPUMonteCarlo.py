import pandas as pd
import numpy as np
from cuml.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

arquivos = ["20150101.txt", "20150102.txt"]

variaveis = ["Duracao", "Servico", "Bytes_origem", "Bytes_destino", "Qtd","Tx_msm_servico", "Tx_Serro", "Tx_Serro_servico", "Destino_qtd_host","Destino_host_qtd_servico", "Destino_host_msm_tx_porta_origem","Destino_host_tx_serro", "Destino_host_tx_serro_servico", "Flag","Detec_IDS", "Detec_Malw", "Detec_Ashula", "Rotulo", "IP_Origem","Porta_Origem", "IP_Destino", "Porta_Destino", "T_Comeco", "Protocolo"]

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

X = X.astype(np.float64)
y = y.astype(np.float32)

n = 100
seed = 895769

tExec = []
acuracias = []
mediasT = []
mediasAc = []

for i in range(n):
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=seed + i)

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)

    inicio = time.time()
    modeloSVM = SVC(kernel='poly', C=1)
    modeloSVM.fit(X_treino_scaled, y_treino)
    fim = time.time()

    tempo_iteracao = fim - inicio
    tExec.append(tempo_iteracao)

    previsao = modeloSVM.predict(X_teste_scaled)
    acuracia = accuracy_score(y_teste, previsao)
    acuracias.append(acuracia)

    mediasT.append(np.mean(tExec))
    mediasAc.append(np.mean(acuracias))

resultados = pd.DataFrame({
    'ite': range(1, n + 1),
    'tExecS': tExec,
    'MediaTCumulativa': mediasT,
    'Acuracia': acuracias,
    'MediaAcCumulativa': mediasAc
})

resultados.to_csv('resultados_monte_carlo_polinomial.csv', index=False)