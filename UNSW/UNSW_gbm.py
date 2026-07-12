import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
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

# Otimizações para GPU
X = X.astype(np.float32)
y = y.astype(np.float32)

X_treino = X[X['is_train'] == 1].drop('is_train', axis=1)
X_teste  = X[X['is_train'] == 0].drop('is_train', axis=1)
y_treino = y[X['is_train'] == 1]
y_teste  = y[X['is_train'] == 0]

scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

dados_treino = lgb.Dataset(X_treino_scaled, label=y_treino)
dados_teste = lgb.Dataset(X_teste_scaled, label=y_teste, reference=dados_treino)

params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.4,
    'num_leaves': 50,
    'max_depth': -1,
    'device_type': 'gpu',
    'gpu_use_dp': False
}

inicio = time.time()
modelo = lgb.train(
    params,
    dados_treino,
    num_boost_round=500,
    valid_sets=[dados_treino, dados_teste],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)
fim = time.time()

probabilidades = modelo.predict(X_teste_scaled, num_iteration=modelo.best_iteration)
y_pred = np.where(probabilidades > 0.5, 1, 0)

print(f"\ntempo de treino: {fim - inicio} segundos\n")
acuracia = accuracy_score(y_teste, y_pred)
matriz_confusao = confusion_matrix(y_teste, y_pred)

df1 = pd.DataFrame(matriz_confusao,
                   index=["Normal (0)", "Ataque (1)"],
                   columns=["Previsao Normal", "Previsao Ataque"])

print(f"Acurácia: {acuracia * 100:.2f}%")
print("Matriz de Confusão:")
print(df1)