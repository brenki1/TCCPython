import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import time

variaveis = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
             "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
             "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
             "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
             "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
             "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
             "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
             "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
             "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
             "label", "difficulty"]

train_df = pd.read_csv("nslkdd/KDDTrain+_20Percent.txt", header=None, names=variaveis)
test_df = pd.read_csv("nslkdd/KDDTest+.txt", header=None, names=variaveis)

train_df['is_train'] = 1
test_df['is_train'] = 0
nsl = pd.concat([train_df, test_df], ignore_index=True)

filtro = ["label", "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
          "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
          "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
          "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
          "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
          "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
          "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
          "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
          "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "is_train"]

nslFiltrada = nsl[filtro].dropna().copy()

nslFiltrada['label'] = nslFiltrada['label'].apply(lambda x: 0 if x == 'normal' else 1)

y = nslFiltrada['label']
X = nslFiltrada.drop(['label', 'is_train'], axis=1)

X = pd.get_dummies(X, columns=['protocol_type', 'service', 'flag'], drop_first=True)

X = X.astype(np.float32)
y = y.astype(np.int32)

X_treino = X[nslFiltrada['is_train'] == 1]
X_teste = X[nslFiltrada['is_train'] == 0]
y_treino = y[nslFiltrada['is_train'] == 1]
y_teste = y[nslFiltrada['is_train'] == 0]

scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

dados_treino = xgb.DMatrix(X_treino_scaled, label=y_treino)
dados_teste = xgb.DMatrix(X_teste_scaled, label=y_teste)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error',
    'learning_rate': 0.5,
    'max_depth': 8,
    'tree_method': 'hist',
    'nthread': 16
}

inicio = time.time()
modelo = xgb.train(
    params,
    dados_treino,
    num_boost_round=500,
    evals=[(dados_treino, 'treino'), (dados_teste, 'teste')],
    early_stopping_rounds=50,
    verbose_eval=False
)
fim = time.time()

probabilidades = modelo.predict(dados_teste, iteration_range=(0, modelo.best_iteration + 1))
y_pred = np.where(probabilidades > 0.5, 1, 0)

print(f"tempo de treino: {fim - inicio} segundos\n")
acuracia = accuracy_score(y_teste, y_pred)
matriz_confusao = confusion_matrix(y_teste, y_pred)

df1 = pd.DataFrame(matriz_confusao,
                   index=["Normal (0)", "Ataque (1)"],
                   columns=["Previsao Normal", "Previsao Ataque"])

print(f"Acurácia: {acuracia * 100:.2f}%")
print("Matriz de Confusão:")
print(df1)