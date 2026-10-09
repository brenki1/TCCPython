import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

diretorio_atual = Path(__file__).resolve().parent
caminho_treino = diretorio_atual.parent.parent / "UNSW" / "UNSW_NB15_training-set.csv"
caminho_teste = diretorio_atual.parent.parent / "UNSW" / "UNSW_NB15_testing-set.csv"

df_treino = pd.read_csv(caminho_treino)
df_teste = pd.read_csv(caminho_teste)

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
X_teste = X[X['is_train'] == 0].drop('is_train', axis=1)
y_treino = y[X['is_train'] == 1]
y_teste = y[X['is_train'] == 0]

scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

X_treino_tensor = torch.tensor(X_treino_scaled).unsqueeze(1)
y_treino_tensor = torch.tensor(y_treino.values).unsqueeze(1)
X_teste_tensor = torch.tensor(X_teste_scaled).unsqueeze(1)
y_teste_tensor = torch.tensor(y_teste.values).unsqueeze(1)

ds_treino = TensorDataset(X_treino_tensor, y_treino_tensor)
ds_teste = TensorDataset(X_teste_tensor, y_teste_tensor)

train_loader = DataLoader(ds_treino, batch_size=4096, shuffle=True, pin_memory=True)
test_loader = DataLoader(ds_teste, batch_size=4096, shuffle=False, pin_memory=True)


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

peso_positivos = (y_treino == 0).sum() / (y_treino == 1).sum()
pos_weight = torch.tensor([peso_positivos], dtype=torch.float32).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(modelo.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

scaler_amp = torch.amp.GradScaler('cuda')

epochs = 50
patience = 8
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