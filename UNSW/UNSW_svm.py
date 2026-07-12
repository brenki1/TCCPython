from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import time

treino = "UNSW_NB15_training-set.csv"
teste = "UNSW_NB15_testing-set.csv"

df_treino = pd.read_csv(treino)
df_teste = pd.read_csv(teste)

df_treino['is_train'] = 1
df_teste['is_train'] = 0

unsw = pd.concat([df_treino, df_teste], ignore_index=True)

if 'label' in unsw.columns:
    unsw.rename(columns={'label': 'Label'}, inplace=True)

colunas_remover = ["id", "srcip", "sport", "dstip", "dsport", "attack_cat", "Stime", "Ltime"]
colunas_existentes = [col for col in colunas_remover if col in unsw.columns]
unsw = unsw.drop(columns=colunas_existentes, errors='ignore')

y = unsw['Label']
X = unsw.drop('Label', axis=1)

cols_categoricas = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=cols_categoricas, drop_first=True)

X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

X = X.astype(np.float32)
y = y.astype(np.float32)

X_treino = X[X['is_train'] == 1].drop('is_train', axis=1)
X_teste  = X[X['is_train'] == 0].drop('is_train', axis=1)
y_treino = y[X['is_train'] == 1]
y_teste  = y[X['is_train'] == 0]

scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

inicio = time.time()
modelo = SVC(kernel='rbf', C=0.31793286660846654, gamma=0.1, random_state=123)
modelo.fit(X_treino_scaled, y_treino)
fim = time.time()

y_pred = modelo.predict(X_teste_scaled)

print(f"\ntempo de treino: {fim - inicio} segundos\n")
acuracia = accuracy_score(y_teste, y_pred)
matriz_confusao = confusion_matrix(y_teste, y_pred)

df = pd.DataFrame(matriz_confusao,
                  index=["Normal (0)", "Ataque (1)"],
                  columns=["Previsao Normal", "Previsao Ataque"])

print(f"Acurácia: {acuracia * 100:.2f}%")
print("Matriz de Confusão:")
print(df)