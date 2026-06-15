from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
import optuna
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Procedimentos que ja eram feitos no R, traduzidos para o Python
arquivos = ["20150101.txt"]

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

# Transforma em flags de True ou False (0 1)
X = pd.get_dummies(X, columns=['Servico', 'Flag', 'Protocolo'], drop_first=True)

# Otimizações para GPU
X = X.astype(np.float32)
y = y.astype(np.float32)

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=895769)

# Funciona como uma espécie de ajuste de importância
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# sem otimização
print("\n")

inicio_base = time.time()
modeloSVM_base = SVC(kernel='poly', C=1)
modeloSVM_base.fit(X_treino_scaled, y_treino)
fim_base = time.time()

print(f"tempo de treino: {fim_base - inicio_base} segundos\n")

previsao_base = modeloSVM_base.predict(X_teste_scaled)
acuracia_base = accuracy_score(y_teste, previsao_base)
matriz_base = confusion_matrix(y_teste, previsao_base)

print("Matriz de confusão")
df_base = pd.DataFrame(
    matriz_base,
    index=["Normal (0)", "Ataque (1)"],
    columns=["Previsao Normal", "Previsao Ataque"]
)

print(df_base)
print(f"\n Acurácia: {acuracia_base}")

#Com otimização
print("\n")
print("Resultados com otimização")

def objective(trial):
    c = trial.suggest_float('C', 1e-3, 10.0, log=True)
    grau = trial.suggest_int('degree', 2, 5)
    coef0 = trial.suggest_float('coef0', 0.0, 10.0)

    modelo = SVC(kernel='poly', C=c, degree=grau, coef0=coef0)

    score = cross_val_score(modelo, X_treino_scaled, y_treino, cv=3, scoring='accuracy').mean()
    return score

estudo = optuna.create_study(direction='maximize')
estudo.optimize(objective, n_trials=10)

print(f"\nMelhores parâmetros encontrados pelo Optuna: {estudo.best_params}\n")

inicio_opt = time.time()
modeloSVM_opt = SVC(kernel='poly', **estudo.best_params)
modeloSVM_opt.fit(X_treino_scaled, y_treino)
fim_opt = time.time()

print(f"tempo de treino: {fim_opt - inicio_opt} segundos\n")

previsao_opt = modeloSVM_opt.predict(X_teste_scaled)
acuracia_opt = accuracy_score(y_teste, previsao_opt)
matriz_opt = confusion_matrix(y_teste, previsao_opt)

print("Matriz de confusão")
df_opt = pd.DataFrame(
    matriz_opt,
    index=["Normal (0)", "Ataque (1)"],
    columns=["Previsao Normal", "Previsao Ataque"]
)

print(df_opt)
print(f"\n Acurácia: {acuracia_opt}")