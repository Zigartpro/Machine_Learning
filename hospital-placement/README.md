# 🏥 Ubicación Óptima de Hospitales — Optimización con ML

Dashboard interactivo para calcular la ubicación óptima de hospitales en una cuadrícula urbana, minimizando la distancia promedio entre casas y centros de salud. Implementa un algoritmo iterativo inspirado en K-Means.

🔗 **Demo en vivo:** [hospitales-rr81.onrender.com](https://hospitales-rr81.onrender.com/)

---

## 🧠 ¿Cómo funciona?

1. El usuario define una cuadrícula (filas × columnas), el número de casas y de hospitales a ubicar.
2. El sistema genera posiciones aleatorias para las casas en la cuadrícula.
3. El algoritmo de optimización itera para encontrar las posiciones de los hospitales que **minimizan la distancia total** entre cada casa y su hospital más cercano.
4. El resultado se visualiza en tiempo real sobre el dashboard.

---

## ✨ Características

- Configuración dinámica de parámetros (tamaño de grilla, casas, hospitales, iteraciones)
- Visualización interactiva del proceso de optimización
- Algoritmo basado en asignación por proximidad + recentrado iterativo
- Interfaz web desplegada en Render

---

## 🛠 Stack tecnológico

| Capa | Tecnología |
|---|---|
| Backend | Python · Flask |
| Algoritmo | NumPy · Lógica personalizada (K-Means inspirado) |
| Frontend | HTML · CSS · JavaScript (Canvas / visualización) |
| Deploy | Render |

---

## 🧮 Algoritmo de optimización

```
1. Inicializar k hospitales en posiciones aleatorias de la grilla
2. Para cada iteración:
   a. Asignar cada casa al hospital más cercano (distancia euclidiana)
   b. Recalcular la posición de cada hospital como el centroide de sus casas asignadas
3. Repetir hasta convergencia o max_iters
4. Retornar posiciones finales óptimas
```

Este enfoque es análogo al algoritmo **K-Means** aplicado a un problema de optimización espacial.

---

## 🚀 Ejecutar localmente

```bash
git clone https://github.com/Zigartpro/Machine_Learning.git
cd Machine_Learning/hospital-placement
pip install -r requirements.txt
python app.py
```

Accede en `http://localhost:5000`

---

## 📌 Conceptos aplicados

- Algoritmos de clustering y optimización espacial
- Principio de K-Means adaptado a ubicación de servicios
- Visualización de algoritmos iterativos en tiempo real
- Despliegue de aplicaciones Python en producción

---

## 👤 Autor

**Duvan Federico Sarmiento Lugo**  
📧 lugosarmiento7@gmail.com | 🔗 [GitHub](https://github.com/Zigartpro)
