import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

# 1. Cargar dataset
df = pd.read_csv("dataset_correos_mejorado.csv")

# 2. Preparar variable objetivo (ham=0, spam=1)
df["Categoria"] = df["Categoria"].map({"ham": 0, "spam": 1})

# 3. Seleccionar variables
X = df[["Palabras_Clave", "Longitud_Caracteres", "Dominio_gratuito",
        "Numero_links", "Caracteres_Especiales"]]
y = df["Categoria"]

# 4. Ejecutar 
exactitud = []
f1 = []

for i in range(800):
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=i)
    
    # Crear y entrenar modelo Decision Tree
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    
    # Métricas
    exactitud.append(accuracy_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))

# Calcular Z scores
z_score = zscore(exactitud)

# 5. Evaluación completa del último modelo
from sklearn.metrics import confusion_matrix, classification_report

print("\nEvaluación del último modelo:")
print("Exactitud:", round(accuracy_score(y_test, y_pred), 3))
print("Z Score:", round(z_score[-1], 3))

cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(cm)

report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
print("\nReporte de clasificación:")
print(report)

# PRUEBA CON DATOS NUEVOS
print("PRUEBA")

# Separar datos que NO se usaron en ningún experimento
X_nuevos, _, y_nuevos, _ = train_test_split(X, y, test_size=0.85, random_state=9999)

# Entrenar modelo final
modelo_final = DecisionTreeClassifier(random_state=42)
modelo_final.fit(X_train, y_train)  # Usar último train
y_pred_nuevos = modelo_final.predict(X_nuevos)

print("Con datos completamente nuevos:")
print("Exactitud:", round(accuracy_score(y_nuevos, y_pred_nuevos), 3))
print("Matriz de confusión:")
print(confusion_matrix(y_nuevos, y_pred_nuevos))

# Mostrar 10 predicciones individuales del árbol
print("\n10 Predicciones individuales del árbol:")
print("-" * 50)
for i in range(10):
    real = "Ham" if y_nuevos.iloc[i] == 0 else "Spam"
    pred = "Ham" if y_pred_nuevos[i] == 0 else "Spam"
    correcto = "✓" if y_nuevos.iloc[i] == y_pred_nuevos[i] else "✗"
    print(f"Correo {i+1}: Real={real:4} | Predicción={pred:4} | {correcto}")

print(f"\nTotal datos nuevos probados: {len(X_nuevos)}")

# 6. Graficar
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, 801), exactitud, 'b-o', markersize=4)
plt.title("Exactitud X Experimento")
plt.xlabel("Experimento")
plt.ylabel("Exactitud")

plt.subplot(1, 3, 2)
plt.plot(range(1, 801), f1, 'g-s', markersize=4)
plt.title("F1 Score X Experimento")
plt.xlabel("Experimento")
plt.ylabel("F1 Score")

plt.subplot(1, 3, 3)
plt.plot(range(1, 801), z_score, 'm-^', markersize=4)
plt.title("Z Score X Experimento")
plt.xlabel("Experimento")
plt.ylabel("Z Score")
plt.axhline(0, color="red", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()