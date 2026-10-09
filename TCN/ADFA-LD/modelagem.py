import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
adfa_ld_path = os.path.abspath(os.path.join(diretorio_atual, '..', '..', 'ADFA-LD'))

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

vectorizer = TfidfVectorizer(
    ngram_range=(2, 4),
    max_features=5000,
    analyzer='word',
    sublinear_tf=True
)

X = vectorizer.fit_transform(corpus).toarray()
y = np.array(rotulos)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

X_treino_tensor = torch.tensor(X_treino_scaled, dtype=torch.float32).unsqueeze(1)
y_treino_tensor = torch.tensor(y_treino, dtype=torch.float32).unsqueeze(1)
X_teste_tensor  = torch.tensor(X_teste_scaled, dtype=torch.float32).unsqueeze(1)
y_teste_tensor  = torch.tensor(y_teste, dtype=torch.float32).unsqueeze(1)

ds_treino = TensorDataset(X_treino_tensor, y_treino_tensor)
ds_teste  = TensorDataset(X_teste_tensor, y_teste_tensor)

train_loader = DataLoader(ds_treino, batch_size=256, shuffle=True, pin_memory=True)
test_loader  = DataLoader(ds_teste, batch_size=256, shuffle=False, pin_memory=True)

class TCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(TCNModel, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out = self.tcn(x)
        out = self.pool(out).squeeze(-1)
        out = self.fc(out)
        return out

modelo = TCNModel(input_dim=X_treino_scaled.shape[1]).to(device)

peso_positivo = float(sum(y_treino == 0) / max(sum(y_treino == 1), 1))
pos_weight = torch.tensor([peso_positivo], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(modelo.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
scaler_amp = torch.amp.GradScaler('cuda')

epochs = 50
patience = 10
best_loss = float('inf')
patience_counter = 0
best_model_state = None

inicio = time.time()

for epoch in range(epochs):
    modelo.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = modelo(X_batch)
            loss = criterion(outputs, y_batch)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

    modelo.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in test_loader:
            X_val, y_val = X_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = modelo(X_val)
                val_loss += criterion(outputs, y_val).item()

    val_loss /= len(test_loader)
    scheduler.step(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        best_model_state = modelo.state_dict()
    else:
        patience_counter += 1

    if patience_counter >= patience:
        break

fim = time.time()

if best_model_state:
    modelo.load_state_dict(best_model_state)

modelo.eval()
y_pred_list = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        outputs = modelo(X_batch)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int()
        y_pred_list.extend(preds.cpu().numpy().flatten())

y_pred = np.array(y_pred_list)

acuracia = accuracy_score(y_teste, y_pred)
matriz_confusao = confusion_matrix(y_teste, y_pred)
df1 = pd.DataFrame(matriz_confusao,
                   index=["Normal (0)", "Ataque (1)"],
                   columns=["Previsao Normal", "Previsao Ataque"])

print(f"\nTempo de treino: {fim - inicio:.2f} segundos\n")
print(f"Acurácia: {acuracia * 100:.2f}%")
print("Matriz de Confusão:")
print(df1)