# 🗳️ Predicción de Intención de Voto — Clasificación KNN

Sistema web de predicción de intención de voto basado en el algoritmo **K-Nearest Neighbors (KNN)**. Dado un perfil socioeconómico del votante, el modelo predice su probable intención de voto con un nivel de confianza.

🔗 **Demo en vivo:** [voto-knn.onrender.com](https://voto-knn.onrender.com/)

---

## 🧠 ¿Qué predice el modelo?

A partir de variables socioeconómicas del votante, el sistema clasifica su **intención de voto** y devuelve:

- ✅ Clase predicha (intención de voto)
- 📊 Nivel de confianza (%)
- 🔢 Valor de K óptimo utilizado
- 🎯 Accuracy del modelo
- 📐 Distancia métrica: **Euclidiana**

---

## 📋 Variables de entrada

| Variable | Tipo | Descripción |
|---|---|---|
| Edad | Numérica | Edad del votante |
| Género | Categórica | Masculino / Femenino / Otro |
| Nivel educativo | Ordinal | Sin estudios → Posgrado |
| Estado laboral | Categórica | Empleado, Independiente, Estudiante... |
| Rango de ingresos | Ordinal | 0–1 SM hasta 8+ SM |
| Tipo de zona | Categórica | Rural / Urbano / Periurbano |
| Fuerza partidista | Numérica | Escala 0–4 |
| Votó antes | Binaria | Sí / No |
| Estado civil | Categórica | Soltero, Casado... |
| Tamaño del hogar | Numérica | Número de personas |
| Tiene hijos | Binaria | Sí / No |
| Región | Categórica | Norte, Sur, Centro, Oriente, Occidente |

---

## 🛠 Stack tecnológico

| Capa | Tecnología |
|---|---|
| Backend / API | Python · Flask |
| ML | Scikit-learn (KNeighborsClassifier) |
| Preprocesamiento | Pandas · LabelEncoder / OneHotEncoder |
| Frontend | HTML · CSS · JavaScript (fetch API) |
| Deploy | Render |

---

## ⚙️ Arquitectura

```
Frontend (HTML/JS)
      │
      │ fetch POST /predict
      ▼
Flask API (Python)
      │
      ├── Preprocesamiento de variables
      ├── Carga del modelo KNN entrenado (.pkl)
      └── Retorna: clase predicha + confianza + métricas
```

---

## 🚀 Ejecutar localmente

```bash
git clone https://github.com/Zigartpro/Machine_Learning.git
cd Machine_Learning/proyecto-knn
pip install -r requirements.txt
python app.py
```

Accede en `http://localhost:5000`

---

## 📌 Conceptos aplicados

- Algoritmo KNN para clasificación multiclase
- Selección del K óptimo mediante validación cruzada
- Codificación de variables categóricas y ordinales
- API REST con Flask para servir modelos ML
- Arquitectura frontend-backend desacoplada
- Despliegue de modelos en producción con Render

---

## 👤 Autor

**Duvan Federico Sarmiento Lugo**  
📧 lugosarmiento7@gmail.com | 🔗 [GitHub](https://github.com/Zigartpro)
