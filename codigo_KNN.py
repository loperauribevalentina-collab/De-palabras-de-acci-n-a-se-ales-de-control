# -- coding: utf-8 --
"""
Entrenamiento de modelo KNN con preprocesamiento.
"""

# ============================
# 1. Librerías
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ============================
# 2. Cargar datos
# ============================
df = pd.read_excel(
    r"C:/Users/Isabela Duque/Desktop/Ingeniería Biomédica ITM/SEMESTRE 6°/Automatización II/Entrega Final/base_grande.xlsx"
)

print("Columnas categóricas detectadas:")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(categorical_cols)

print("\nPrimeras filas del dataset:")
print(df.head())

# ============================
# 3. Definir variables
# ============================
X = df.drop("target", axis=1)
y = df["target"]

# Separar columnas numéricas y categóricas
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# ============================
# 4. Separar en train/test
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 5. Preprocesamiento y Pipeline
# ============================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(), categorical_cols)
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("knn", KNeighborsClassifier())
])

# ============================
# 6. Búsqueda de hiperparámetros
# ============================
param_grid = {
    "knn__n_neighbors": list(range(1, 31)),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan", "minkowski"]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nMejores hiperparámetros encontrados:")
print(grid.best_params_)
print("\nMejor score en validación:", grid.best_score_)

# ============================
# 7. Evaluación final
# ============================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
# ============================
# 8. Graficar Matriz de Confusión
# ============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)

# Etiquetas
plt.title("Matriz de Confusión - KNN")
plt.xlabel("Predicción")
plt.ylabel("Real")

# Etiquetas de clases
plt.xticks([0, 1], ["Arm", "Leg"])
plt.yticks([0, 1], ["Arm", "Leg"])

# Mostrar valores dentro de cada celda
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center", color="black", fontsize=12)

plt.tight_layout()
plt.show()
