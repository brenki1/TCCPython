from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import time

arq_treino = "kdd-cup-1999-data/kddcup.data_10_percent/kddcup.data_10_percent"
arq_teste = "kdd-cup-1999-data/kddcup.data.corrected"

variaveis = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

df_treino = pd.read_csv(arq_treino, header=None, names=variaveis).dropna()
df_teste = pd.read_csv(arq_teste, header=None, names=variaveis).dropna()

df_treino['label'] = np.where(df_treino['label'].isin(['normal.', 'normal']), 0, 1)
df_teste['label'] = np.where(df_teste['label'].isin(['normal.', 'normal']), 0, 1)

n_treino = len(df_treino)
df_total = pd.concat([df_treino, df_teste], ignore_index=True)

y_total = df_total['label']
X_total = df_total.drop('label', axis=1)

X_total = pd.get_dummies(X_total, columns=['protocol_type', 'service', 'flag'], drop_first=True)

X_total = X_total.astype(np.float32)
y_total = y_total.astype(np.int32)

X_treino = X_total.iloc[:n_treino]
X_teste = X_total.iloc[n_treino:]
y_treino = y_total.iloc[:n_treino]
y_teste = y_total.iloc[n_treino:]

scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

inicio = time.time()
modelo = LinearSVC(dual=False, random_state=895769, max_iter=1000)
modelo.fit(X_treino_scaled, y_treino)
fim = time.time()

y_pred = modelo.predict(X_teste_scaled)

print(f"tempo de treino: {fim - inicio} segundos\n")
acuracia = accuracy_score(y_teste, y_pred)
matriz_confusao = confusion_matrix(y_teste, y_pred)

df1 = pd.DataFrame(matriz_confusao,
                   index=["Normal (0)", "Ataque (1)"],
                   columns=["Previsao Normal", "Previsao Ataque"])

print(f"Acurácia: {acuracia * 100:.2f}%")
print("Matriz de Confusão:")
print(df1)