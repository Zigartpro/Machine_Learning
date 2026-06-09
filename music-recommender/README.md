# 🎵 MusicIA Colombia — Sistema de Recomendación Musical

Sistema web de recomendación de música colombiana basado en Machine Learning. Analiza las preferencias del usuario y sugiere canciones usando algoritmos de similitud y clasificación con Scikit-learn.

🔗 **Demo en vivo:** [recomendador-mml.onrender.com](https://recomendador-mml.onrender.com/)

---

## 🧠 ¿Cómo funciona?

El sistema recibe características del usuario (género musical preferido, ritmo, estado de ánimo, etc.) y usa un modelo entrenado para encontrar las canciones más similares en el dataset, devolviendo recomendaciones personalizadas en tiempo real.

---

## ✨ Características

- Recomendación personalizada basada en preferencias del usuario
- Modelo ML entrenado con Scikit-learn
- Interfaz web interactiva desplegada en Render
- Enfocado en música colombiana

---

## 🛠 Stack tecnológico

| Capa | Tecnología |
|---|---|
| Backend | Python · Flask |
| ML | Scikit-learn |
| Frontend | HTML · CSS · JavaScript |
| Deploy | Render |

---

## 🚀 Ejecutar localmente

```bash
git clone https://github.com/Zigartpro/Machine_Learning.git
cd Machine_Learning/music-recommender
pip install -r requirements.txt
python app.py
```

Accede en `http://localhost:5000`

---

## 📌 Conceptos aplicados

- Filtrado basado en contenido (Content-Based Filtering)
- Métricas de similitud entre vectores de características
- Serialización de modelos con `joblib`
- Despliegue de modelos ML en producción con Render

---

## 👤 Autor

**Duvan Federico Sarmiento Lugo**  
📧 lugosarmiento7@gmail.com | 🔗 [GitHub](https://github.com/Zigartpro)
