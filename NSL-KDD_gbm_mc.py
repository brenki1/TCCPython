import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
y = y.astype(np.float32)

X_treino = X[nslFiltrada['is_train'] == 1]
X_teste = X[nslFiltrada['is_train'] == 0]
y_treino = y[nslFiltrada['is_train'] == 1]
y_teste = y[nslFiltrada['is_train'] == 0]

scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

dados_treino = lgb.Dataset(X_treino_scaled, label=y_treino)
dados_teste = lgb.Dataset(X_teste_scaled, label=y_teste, reference=dados_treino)

base_params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.5,
    'num_leaves': 50,
    'max_depth': -1,
    'device_type': 'gpu',
    'gpu_use_dp': False,
    'verbose': -1
}

n = 10000

vetor_acuracia = np.empty(n)
vetor_tempo = np.empty(n)

for i in range(n):
    params = base_params.copy()
    params['seed'] = np.random.randint(0, 2147483647)

    inicio = time.time()
    modelo = lgb.train(
        params,
        dados_treino,
        num_boost_round=20,
        valid_sets=[dados_treino, dados_teste],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    fim = time.time()

    vetor_tempo[i] = fim - inicio

    probabilidades = modelo.predict(X_teste_scaled, num_iteration=modelo.best_iteration)
    y_pred = np.where(probabilidades > 0.5, 1, 0)

    vetor_acuracia[i] = accuracy_score(y_teste, y_pred) * 100

media_cumulativa_acuracia = np.cumsum(vetor_acuracia) / np.arange(1, n + 1)
media_cumulativa_tempo = np.cumsum(vetor_tempo) / np.arange(1, n + 1)

df_export = pd.DataFrame({
    'ite': np.arange(1, n + 1),
    'Acuracia': vetor_acuracia,
    'tExecS': vetor_tempo,
    'MediaAcCumulativa': media_cumulativa_acuracia,
    'MediaTCumulativa': media_cumulativa_tempo
})

df_export.to_csv("monteCarloNSLKDDgbm.csv", index=False)