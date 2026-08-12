import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import time

adfa_ld_path = ''

corpus = []
rotulos = []

for pasta in ['Training_Data_Master', 'Validation_Data_Master']:
    caminho_pasta = os.path.join(adfa_ld_path, pasta)
    for arquivo in sorted(os.listdir(caminho_pasta)):
        caminho_arquivo = os.path.join(caminho_pasta, arquivo)
        if os.path.isfile(caminho_arquivo):
            with open(caminho_arquivo, 'r') as f:
                syscalls = ' '.join(f.read().split())
                if syscalls:
                    corpus.append(syscalls)
                    rotulos.append(0)

pasta_ataques = os.path.join(adfa_ld_path, 'Attack_Data_Master')
for tipo_ataque in sorted(os.listdir(pasta_ataques)):
    pasta_tipo = os.path.join(pasta_ataques, tipo_ataque)
    if os.path.isdir(pasta_tipo):
        for arquivo in sorted(os.listdir(pasta_tipo)):
            caminho_arquivo = os.path.join(pasta_tipo, arquivo)
            if os.path.isfile(caminho_arquivo):
                with open(caminho_arquivo, 'r') as f:
                    syscalls = ' '.join(f.read().split())
                    if syscalls:
                        corpus.append(syscalls)
                        rotulos.append(1)

vectorizer = TfidfVectorizer(ngram_range=(2, 4), max_features=5000, analyzer='word', sublinear_tf=True)
X = vectorizer.fit_transform(corpus).toarray().astype(np.float32)
y = np.array(rotulos, dtype=np.float32)

n = 450
resultados = []
soma_acuracia = 0
soma_tempo = 0

for i in range(1, n + 1):
    seed = np.random.randint(0, 1000000)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino).astype(np.float32)
    X_teste_scaled = scaler.transform(X_teste).astype(np.float32)

    dtreino = xgb.DMatrix(X_treino_scaled, label=y_treino)
    dteste = xgb.DMatrix(X_teste_scaled, label=y_teste)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': float(sum(y_treino == 0) / max(sum(y_treino == 1), 1)),
        'nthread': 16,
        'seed': seed,
        'verbosity': 0
    }

    inicio = time.time()
    modelo = xgb.train(
        params,
        dtreino,
        num_boost_round=500,
        evals=[(dtreino, 'treino'), (dteste, 'teste')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    fim = time.time()

    tempo_exec = fim - inicio
    soma_tempo += tempo_exec
    media_tempo_cumulativa = soma_tempo / i

    probabilidades = modelo.predict(dteste, iteration_range=(0, modelo.best_iteration + 1))
    y_pred = np.where(probabilidades > 0.5, 1, 0)

    acuracia = accuracy_score(y_teste, y_pred)
    soma_acuracia += acuracia
    media_acuracia_cumulativa = soma_acuracia / i

    resultados.append({
        'ite': i,
        'Acuracia': acuracia,
        'tExecS': tempo_exec,
        'MediaAcCumulativa': media_acuracia_cumulativa,
        'MediaTCumulativa': media_tempo_cumulativa
    })

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("monteCarloADFA_xgb.csv", index=False)