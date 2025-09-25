from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["target_name"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
print(df)

from sklearn.linear_model import LinearRegression

# 1. Preparar variables
X = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
y = df["target"]

# 2. Crear y cargar modelo
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Predicción
y_pred_cont = lin_reg.predict(X)       
y_pred_class = np.round(y_pred_cont).astype(int) 

# evaluar el modelo
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc = accuracy_score(y, y_pred_class)
cm = confusion_matrix(y, y_pred_class)
report = classification_report(y, y_pred_class, target_names=["setosa", "versicolor", "virginica"])

print("Exactitud del modelo (sobre todos los datos):", acc)
print("\nMatriz de confusión:\n", cm)
print("\nReporte de clasificación:\n", report)

# 3. Prueba con nuevos datos
nuevos_datos = pd.DataFrame([[5.1, 3.5, 1.4, 0.2],
                         [6.7, 3.0, 5.2, 2.3]],
                         columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])

# Predicción
predicciones = lin_reg.predict(nuevos_datos)
print("Los datos dados son de:", data.target_names[np.round(predicciones).astype(int)])

# graficas de barras por promedios de las dimensiones por especie 
import matplotlib.pyplot as plt

# Renombrar columnas al español
df_renombrado = df.rename(columns={
    "sepal length (cm)": "Largo del sépalo (cm)",
    "sepal width (cm)": "Ancho del sépalo (cm)",
    "petal length (cm)": "Largo del pétalo (cm)",
    "petal width (cm)": "Ancho del pétalo (cm)"
})

# Agrupar y calcular promedios SIN la columna target
promedios = df_renombrado.drop(columns=["target"]).groupby("target_name").mean()

# Graficar
promedios.plot(kind="bar", figsize=(10, 6))
plt.title("Promedio de las dimensiones por especie")
plt.ylabel("Promedio (cm)")
plt.xlabel("Especie")
plt.xticks(rotation=0)
plt.legend(title="Características")
plt.show()
