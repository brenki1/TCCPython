import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

dados_treino = xgb.DMatrix(X_treino_scaled, label=y_treino)
dados_teste = xgb.DMatrix(X_teste_scaled, label=y_teste)

n = 1000
resultados = []
soma_acuracia = 0
soma_tempo = 0

for i in range(1, n + 1):
    seed = np.random.randint(0, 1000000)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'error',
        'learning_rate': 0.07223112058646232,
        'max_depth': 4,
        'tree_method': 'hist',
        'nthread': 16,
        'seed': seed
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

    tempo_exec = fim - inicio
    soma_tempo += tempo_exec
    media_tempo_cumulativa = soma_tempo / i

    probabilidades = modelo.predict(dados_teste)
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
df_resultados.to_csv("monteCarloUNSWxgb.csv", index=False)