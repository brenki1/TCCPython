from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
import glob
import random
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

resultados_metricas = []

# Kyoto

variaveis = ["Duracao", "Servico", "Bytes_origem", "Bytes_destino", "Qtd", "Tx_msm_servico", "Tx_Serro",
             "Tx_Serro_servico", "Destino_qtd_host", "Destino_host_qtd_servico", "Destino_host_msm_tx_porta_origem",
             "Destino_host_tx_serro", "Destino_host_tx_serro_servico", "Flag", "Detec_IDS", "Detec_Malw",
             "Detec_Ashula", "Rotulo", "IP_Origem", "Porta_Origem", "IP_Destino", "Porta_Destino", "T_Comeco", "Protocolo"]

caminho_kyoto = "../dias/201207*.txt"
arquivos = glob.glob(caminho_kyoto)
leitura = []

for arq in arquivos:
    dia = pd.read_csv(arq, sep="\t", header=None, names=variaveis)
    qtdAmostras = min(random.randint(10000, 15000), len(dia))
    amostra = dia.sample(n=qtdAmostras, random_state=123)
    leitura.append(amostra)

kyoto = pd.concat(leitura, ignore_index=True)
kyoto = kyoto[kyoto['Rotulo'] != -2]

filtro = ["Rotulo", "Duracao", "Servico", "Bytes_origem", "Bytes_destino", "Qtd", "Destino_qtd_host",
          "Destino_host_qtd_servico", "Destino_host_tx_serro", "Flag", "Protocolo"]
kyotoFiltrada = kyoto[filtro].dropna().copy()
kyotoFiltrada['Rotulo'] = kyotoFiltrada['Rotulo'].replace({-1: 1, 1: 0})

y_kyoto = kyotoFiltrada['Rotulo']
X_kyoto = kyotoFiltrada.drop('Rotulo', axis=1)

X_kyoto = pd.get_dummies(X_kyoto, columns=['Servico', 'Flag', 'Protocolo'], drop_first=True)

X_kyoto = X_kyoto.astype(np.float32)
y_kyoto = y_kyoto.astype(np.float32)

X_treino_k, X_teste_k, y_treino_k, y_teste_k = train_test_split(X_kyoto, y_kyoto, test_size=0.2, random_state=895769)

scaler_k = StandardScaler()
X_treino_scaled_k = scaler_k.fit_transform(X_treino_k)
X_teste_scaled_k = scaler_k.transform(X_teste_k)

modelo_svm_k = SVC(kernel='rbf', probability=True, random_state=123)
modelo_svm_k.fit(X_treino_scaled_k, y_treino_k)
prob_svm_k = modelo_svm_k.predict_proba(X_teste_scaled_k)[:, 1]
pred_svm_k = np.where(prob_svm_k > 0.5, 1, 0)

dtreino_k = xgb.DMatrix(X_treino_scaled_k, label=y_treino_k)
dteste_k = xgb.DMatrix(X_teste_scaled_k, label=y_teste_k)
params_xgb_k = {'objective': 'binary:logistic', 'eval_metric': 'error', 'learning_rate': 0.1, 'max_depth': 4, 'tree_method': 'hist', 'nthread': 16}
modelo_xgb_k = xgb.train(params_xgb_k, dtreino_k, num_boost_round=500, evals=[(dtreino_k, 'treino'), (dteste_k, 'teste')], early_stopping_rounds=50, verbose_eval=False)
prob_xgb_k = modelo_xgb_k.predict(dteste_k, iteration_range=(0, modelo_xgb_k.best_iteration + 1))
pred_xgb_k = np.where(prob_xgb_k > 0.5, 1, 0)

dados_treino_k = lgb.Dataset(X_treino_scaled_k, label=y_treino_k)
dados_teste_k = lgb.Dataset(X_teste_scaled_k, label=y_teste_k, reference=dados_treino_k)
params_lgb_k = {'objective': 'binary', 'metric': 'binary_error', 'learning_rate': 0.1, 'num_leaves': 50, 'max_depth': -1, 'device_type': 'gpu', 'gpu_use_dp': False}
modelo_lgbm_k = lgb.train(params_lgb_k, dados_treino_k, num_boost_round=500, valid_sets=[dados_treino_k, dados_teste_k], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
prob_lgbm_k = modelo_lgbm_k.predict(X_teste_scaled_k, num_iteration=modelo_lgbm_k.best_iteration)
pred_lgbm_k = np.where(prob_lgbm_k > 0.5, 1, 0)

df_probs_kyoto = pd.DataFrame({'y_true': y_teste_k, 'prob_svm': prob_svm_k, 'prob_xgb': prob_xgb_k, 'prob_lgbm': prob_lgbm_k})
df_probs_kyoto.to_csv("kyotoROC.csv", index=False)

for modelo, pred in [('SVM', pred_svm_k), ('XGBoost', pred_xgb_k), ('LightGBM', pred_lgbm_k)]:
    cm = confusion_matrix(y_teste_k, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    resultados_metricas.append({
        'Dataset': 'Kyoto', 'Modelo': modelo,
        'Acuracia': accuracy_score(y_teste_k, pred),
        'Precisao': precision_score(y_teste_k, pred, zero_division=0),
        'Recall': recall_score(y_teste_k, pred, zero_division=0),
        'F1_Score': f1_score(y_teste_k, pred, zero_division=0),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    })


# UNSW

treino_unsw = "../UNSW/UNSW_NB15_training-set.csv"
teste_unsw = "../UNSW/UNSW_NB15_testing-set.csv"

df_treino_u = pd.read_csv(treino_unsw)
df_teste_u = pd.read_csv(teste_unsw)

df_treino_u['is_train'] = 1
df_teste_u['is_train'] = 0

unsw = pd.concat([df_treino_u, df_teste_u], ignore_index=True)

if 'label' in unsw.columns:
    unsw.rename(columns={'label': 'Label'}, inplace=True)

colunas_remover_u = ["id", "srcip", "sport", "dstip", "dsport", "attack_cat", "Stime", "Ltime"]
colunas_existentes_u = [col for col in colunas_remover_u if col in unsw.columns]
unsw = unsw.drop(columns=colunas_existentes_u, errors='ignore')

y_unsw = unsw['Label']
X_unsw = unsw.drop('Label', axis=1)

cols_categoricas_u = X_unsw.select_dtypes(include=['object']).columns.tolist()
X_unsw = pd.get_dummies(X_unsw, columns=cols_categoricas_u, drop_first=True)

X_unsw = X_unsw.apply(pd.to_numeric, errors='coerce').fillna(0)

X_unsw = X_unsw.astype(np.float32)
y_unsw = y_unsw.astype(np.float32)

X_treino_u = X_unsw[X_unsw['is_train'] == 1].drop('is_train', axis=1)
X_teste_u  = X_unsw[X_unsw['is_train'] == 0].drop('is_train', axis=1)
y_treino_u = y_unsw[X_unsw['is_train'] == 1]
y_teste_u  = y_unsw[X_unsw['is_train'] == 0]

scaler_u = StandardScaler()
X_treino_scaled_u = scaler_u.fit_transform(X_treino_u)
X_teste_scaled_u = scaler_u.transform(X_teste_u)

modelo_svm_u = SVC(kernel='rbf', C=0.31793286660846654, gamma=0.1, probability=True, random_state=123)
modelo_svm_u.fit(X_treino_scaled_u, y_treino_u)
prob_svm_u = modelo_svm_u.predict_proba(X_teste_scaled_u)[:, 1]
pred_svm_u = np.where(prob_svm_u > 0.5, 1, 0)

dtreino_u = xgb.DMatrix(X_treino_scaled_u, label=y_treino_u)
dteste_u = xgb.DMatrix(X_teste_scaled_u, label=y_teste_u)
params_xgb_u = {'objective': 'binary:logistic', 'eval_metric': 'error', 'learning_rate': 0.07223112058646232, 'max_depth': 4, 'tree_method': 'hist', 'nthread': 16}
modelo_xgb_u = xgb.train(params_xgb_u, dtreino_u, num_boost_round=500, evals=[(dtreino_u, 'treino'), (dteste_u, 'teste')], early_stopping_rounds=50, verbose_eval=False)
prob_xgb_u = modelo_xgb_u.predict(dteste_u, iteration_range=(0, modelo_xgb_u.best_iteration + 1))
pred_xgb_u = np.where(prob_xgb_u > 0.5, 1, 0)

dados_treino_u = lgb.Dataset(X_treino_scaled_u, label=y_treino_u)
dados_teste_u = lgb.Dataset(X_teste_scaled_u, label=y_teste_u, reference=dados_treino_u)
params_lgb_u = {'objective': 'binary', 'metric': 'binary_error', 'learning_rate': 0.4, 'num_leaves': 50, 'max_depth': -1, 'device_type': 'gpu', 'gpu_use_dp': False}
modelo_lgbm_u = lgb.train(params_lgb_u, dados_treino_u, num_boost_round=500, valid_sets=[dados_treino_u, dados_teste_u], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
prob_lgbm_u = modelo_lgbm_u.predict(X_teste_scaled_u, num_iteration=modelo_lgbm_u.best_iteration)
pred_lgbm_u = np.where(prob_lgbm_u > 0.5, 1, 0)

df_probs_unsw = pd.DataFrame({'y_true': y_teste_u, 'prob_svm': prob_svm_u, 'prob_xgb': prob_xgb_u, 'prob_lgbm': prob_lgbm_u})
df_probs_unsw.to_csv("unswROC.csv", index=False)

for modelo, pred in [('SVM', pred_svm_u), ('XGBoost', pred_xgb_u), ('LightGBM', pred_lgbm_u)]:
    cm = confusion_matrix(y_teste_u, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    resultados_metricas.append({
        'Dataset': 'UNSW', 'Modelo': modelo,
        'Acuracia': accuracy_score(y_teste_u, pred),
        'Precisao': precision_score(y_teste_u, pred, zero_division=0),
        'Recall': recall_score(y_teste_u, pred, zero_division=0),
        'F1_Score': f1_score(y_teste_u, pred, zero_division=0),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    })


# ADFA-LD

adfa_ld_path = '../ADFA-LD'

corpus = []
rotulos = []

for pasta in ['Training_Data_Master', 'Validation_Data_Master']:
    caminho_pasta = os.path.join(adfa_ld_path, pasta)
    if os.path.exists(caminho_pasta):
        for arquivo in sorted(os.listdir(caminho_pasta)):
            caminho_arquivo = os.path.join(caminho_pasta, arquivo)
            if os.path.isfile(caminho_arquivo):
                with open(caminho_arquivo, 'r') as f:
                    syscalls = ' '.join(f.read().split())
                    if syscalls:
                        corpus.append(syscalls)
                        rotulos.append(0)

pasta_ataques = os.path.join(adfa_ld_path, 'Attack_Data_Master')
if os.path.exists(pasta_ataques):
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
X_a = vectorizer.fit_transform(corpus).toarray()
y_a = np.array(rotulos)

X_treino_a, X_teste_a, y_treino_a, y_teste_a = train_test_split(X_a, y_a, test_size=0.3, random_state=42, stratify=y_a)

scaler_a = StandardScaler()
X_treino_scaled_a = scaler_a.fit_transform(X_treino_a).astype(np.float32)
X_teste_scaled_a = scaler_a.transform(X_teste_a).astype(np.float32)

y_treino_a = y_treino_a.astype(np.float32)
y_teste_a_f = y_teste_a.astype(np.float32)

peso_positivo_a = float(sum(y_treino_a == 0) / max(sum(y_treino_a == 1), 1))

modelo_svm_a = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True, random_state=42)
modelo_svm_a.fit(X_treino_scaled_a, y_treino_a)
prob_svm_a = modelo_svm_a.predict_proba(X_teste_scaled_a)[:, 1]
pred_svm_a = np.where(prob_svm_a > 0.5, 1, 0)

dtreino_a = xgb.DMatrix(X_treino_scaled_a, label=y_treino_a)
dteste_a = xgb.DMatrix(X_teste_scaled_a, label=y_teste_a_f)
params_xgb_a = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'max_depth': 6, 'learning_rate': 0.5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'scale_pos_weight': peso_positivo_a, 'nthread': 16}
modelo_xgb_a = xgb.train(params_xgb_a, dtreino_a, num_boost_round=500, evals=[(dtreino_a, 'treino'), (dteste_a, 'teste')], early_stopping_rounds=50, verbose_eval=False)
prob_xgb_a = modelo_xgb_a.predict(dteste_a, iteration_range=(0, modelo_xgb_a.best_iteration + 1))
pred_xgb_a = np.where(prob_xgb_a > 0.5, 1, 0)

dados_treino_a = lgb.Dataset(X_treino_scaled_a, label=y_treino_a)
dados_teste_a = lgb.Dataset(X_teste_scaled_a, label=y_teste_a_f, reference=dados_treino_a)
params_lgbm_a = {'objective': 'binary', 'metric': 'binary_error', 'learning_rate': 0.5, 'num_leaves': 63, 'max_depth': -1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'is_unbalance': True, 'device_type': 'gpu', 'gpu_use_dp': False}
modelo_lgbm_a = lgb.train(params_lgbm_a, dados_treino_a, num_boost_round=500, valid_sets=[dados_treino_a, dados_teste_a], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
prob_lgbm_a = modelo_lgbm_a.predict(X_teste_scaled_a, num_iteration=modelo_lgbm_a.best_iteration)
pred_lgbm_a = np.where(prob_lgbm_a > 0.5, 1, 0)

df_probs_adfa = pd.DataFrame({'y_true': y_teste_a_f, 'prob_svm': prob_svm_a, 'prob_xgb': prob_xgb_a, 'prob_lgbm': prob_lgbm_a})
df_probs_adfa.to_csv("adfaROC.csv", index=False)

for modelo, pred in [('SVM', pred_svm_a), ('XGBoost', pred_xgb_a), ('LightGBM', pred_lgbm_a)]:
    cm = confusion_matrix(y_teste_a_f, pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    resultados_metricas.append({
        'Dataset': 'ADFA-LD', 'Modelo': modelo,
        'Acuracia': accuracy_score(y_teste_a_f, pred),
        'Precisao': precision_score(y_teste_a_f, pred, zero_division=0),
        'Recall': recall_score(y_teste_a_f, pred, zero_division=0),
        'F1_Score': f1_score(y_teste_a_f, pred, zero_division=0),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    })

if resultados_metricas:
    df_metricas = pd.DataFrame(resultados_metricas)
    df_metricas.to_csv("metricas.csv", index=False)