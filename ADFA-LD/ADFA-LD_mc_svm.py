from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

n = 1000
resultados = []
soma_acuracia = 0
soma_tempo = 0

for i in range(1, n + 1):
    seed = np.random.randint(0, 1000000)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)

    inicio = time.time()
    modelo = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=seed)
    modelo.fit(X_treino_scaled, y_treino)
    fim = time.time()

    tempo_exec = fim - inicio
    soma_tempo += tempo_exec
    media_tempo_cumulativa = soma_tempo / i

    y_pred = modelo.predict(X_teste_scaled)

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
df_resultados.to_csv("monteCarloADFA_svm.csv", index=False)