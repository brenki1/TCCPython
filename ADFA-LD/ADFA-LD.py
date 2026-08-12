from sklearnex import patch_sklearn
patch_sklearn()

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
import os
import time
import warnings
warnings.filterwarnings('ignore')

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

vectorizer = TfidfVectorizer(
    ngram_range=(2, 4),      # bigramas, trigramas e 4-gramas
    max_features=5000,       # 5000 features mais relevantes
    analyzer='word',
    sublinear_tf=True        # aplica log(1 + tf)
)

X = vectorizer.fit_transform(corpus).toarray()
y = np.array(rotulos)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# Otimização para GPU
X_treino_scaled = X_treino_scaled.astype(np.float32)
X_teste_scaled = X_teste_scaled.astype(np.float32)
y_treino = y_treino.astype(np.float32)
y_teste_f = y_teste.astype(np.float32)

# Peso para classes desbalanceadas
peso_positivo = float(sum(y_treino == 0) / max(sum(y_treino == 1), 1))

print("\n")
print("SVM")

inicio = time.time()
modelo_svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)
modelo_svm.fit(X_treino_scaled, y_treino)
fim = time.time()
tempo_svm = fim - inicio

probabilidades_svm = modelo_svm.predict_proba(X_teste_scaled)[:, 1]
y_pred_svm = np.where(probabilidades_svm > 0.5, 1, 0)

print(f"\nTempo de treino: {tempo_svm:.2f} segundos")

acuracia_svm = accuracy_score(y_teste, y_pred_svm)
cm_svm = confusion_matrix(y_teste, y_pred_svm)

df_cm_svm = pd.DataFrame(cm_svm,
    index=["Normal (0)", "Ataque (1)"],
    columns=["Previsao Normal", "Previsao Ataque"])

print(f"Acurácia:  {acuracia_svm * 100:.2f}%")
print("Matriz de Confusão:")
print(df_cm_svm)

print("\n")
print("XGBoost")

dtreino = xgb.DMatrix(X_treino_scaled, label=y_treino)
dteste = xgb.DMatrix(X_teste_scaled, label=y_teste_f)

params_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': peso_positivo,
    'nthread': 16,
}

inicio = time.time()
modelo_xgb = xgb.train(
    params_xgb,
    dtreino,
    num_boost_round=500,
    evals=[(dtreino, 'treino'), (dteste, 'teste')],
    early_stopping_rounds=50,
    verbose_eval=100
)
fim = time.time()
tempo_xgb = fim - inicio

probabilidades_xgb = modelo_xgb.predict(
    dteste, iteration_range=(0, modelo_xgb.best_iteration + 1)
)
y_pred_xgb = np.where(probabilidades_xgb > 0.5, 1, 0)

print(f"\nTempo de treino: {tempo_xgb:.2f} segundos")

acuracia_xgb = accuracy_score(y_teste, y_pred_xgb)
cm_xgb = confusion_matrix(y_teste, y_pred_xgb)

df_cm_xgb = pd.DataFrame(cm_xgb,
    index=["Normal (0)", "Ataque (1)"],
    columns=["Previsao Normal", "Previsao Ataque"])
print(f"Acurácia:  {acuracia_xgb * 100:.2f}%")
print("Matriz de Confusão:")
print(df_cm_xgb)


print("\n")
print("LightGBM")

dados_treino = lgb.Dataset(X_treino_scaled, label=y_treino)
dados_teste = lgb.Dataset(X_teste_scaled, label=y_teste_f, reference=dados_treino)

params_lgbm = {
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.5,
    'num_leaves': 63,
    'max_depth': -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'is_unbalance': True,
    'device_type': 'gpu',
    'gpu_use_dp': False
}

inicio = time.time()
modelo_lgbm = lgb.train(
    params_lgbm,
    dados_treino,
    num_boost_round=500,
    valid_sets=[dados_treino, dados_teste],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)
fim = time.time()
tempo_lgbm = fim - inicio

probabilidades_lgbm = modelo_lgbm.predict(
    X_teste_scaled, num_iteration=modelo_lgbm.best_iteration
)
y_pred_lgbm = np.where(probabilidades_lgbm > 0.5, 1, 0)

print(f"\nTempo de treino: {tempo_lgbm:.2f} segundos")

acuracia_lgbm = accuracy_score(y_teste, y_pred_lgbm)
cm_lgbm = confusion_matrix(y_teste, y_pred_lgbm)

df_cm_lgbm = pd.DataFrame(cm_lgbm,
    index=["Normal (0)", "Ataque (1)"],
    columns=["Previsao Normal", "Previsao Ataque"])

print(f"Acurácia:  {acuracia_lgbm * 100:.2f}%")
print("Matriz de Confusão:")
print(df_cm_lgbm)