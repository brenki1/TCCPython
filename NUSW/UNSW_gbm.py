import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import time

treino = "UNSW_NB15_training-set.csv"
teste = "UNSW_NB15_testing-set.csv"



variaveis = ["srcip","sport","dstip","dsport",
"proto","state","dur","sbytes","dbytes","sttl",
"dttl","sloss","dloss","service","Sload","Dload","Spkts","Dpkts",
"swin","dwin","stcpb","dtcpb","smeansz","dmeansz","trans_depth","res_bdy_len","Sjit","Djit",
"Stime","Ltime","Sintpkt","Dintpkt","tcprtt",
"synack","ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd","is_ftp_login",
"ct_ftp_cmd","ct_srv_src","ct_srv_dst","ct_dst_ltm",
"ct_src_ ltm","ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm", "attack_cat","Label"]

df_treino = pd.read_csv(treino, header=None, names=variaveis)
df_teste = pd.read_csv

