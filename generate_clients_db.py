"""
generate_clients_db.py
Ejecución única: descarga el dataset UCI Polish Companies Bankruptcy,
aplica el mismo preprocesamiento del notebook y guarda el test set
como clients.csv con IDs tipo CLI-XXXXX.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo

print("Descargando dataset Polish Companies Bankruptcy (UCI id=365)...")
dataset = fetch_ucirepo(id=365)
X = dataset.data.features
y = dataset.data.targets.iloc[:, 0]
print(f"Shape original: {X.shape}")

# Imputar nulos con mediana (igual que notebook)
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Clip de outliers extremos p1-p99 (igual que notebook)
for col in X_imputed.columns:
    q01 = X_imputed[col].quantile(0.01)
    q99 = X_imputed[col].quantile(0.99)
    X_imputed[col] = X_imputed[col].clip(q01, q99)

# Split 80/20 estratificado con mismo random_state que notebook
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
print(f"Test set: {X_test.shape[0]} empresas ({y_test.sum()} quiebras)")

# Asignar IDs secuenciales CLI-XXXXX
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
ids = [f"CLI-{i+1:05d}" for i in range(len(X_test))]

clients = X_test.copy()
clients.insert(0, "id", ids)
clients["class"] = y_test.values

clients.to_csv("clients.csv", index=False)
print(f"clients.csv guardado: {len(clients)} filas, {clients.shape[1]} columnas")
print(f"Rango de IDs: CLI-00001 a CLI-{len(clients):05d}")
