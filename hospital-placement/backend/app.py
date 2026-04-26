from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
# Estos módulos deben estar en tu carpeta junto a app.py
from utils import generate_houses, plot_map
from placer import find_hospital_positions

app = Flask(__name__)
CORS(app)

# --- NUEVAS RUTAS PARA EVITAR EL ERROR 404 ---

@app.route("/", methods=["GET"])
def index():
    """Ruta raíz para verificar que el servidor está vivo"""
    return jsonify({
        "status": "online",
        "message": "Backend de Localización de Hospitales activo",
        "endpoints_disponibles": ["/generate", "/place", "/health"]
    })

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de salud para Render"""
    return jsonify({"status": "healthy"})

# --- ENDPOINTS EXISTENTES ---

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    try:
        m = int(data.get("m", 100))
        n = int(data.get("n", 100))
        num_houses = int(data.get("num_houses", 50))
        houses = generate_houses(m, n, num_houses).tolist()

        can_draw = (m <= 120 and n <= 120)
        return jsonify({"houses": houses, "can_draw": can_draw})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/place", methods=["POST"])
def place():
    data = request.get_json()
    try:
        m = int(data.get("m", 100))
        n = int(data.get("n", 100))
        houses = np.array(data.get("houses", []))
        k = int(data.get("k", 3))
        max_iters = int(data.get("max_iters", 50))

        if len(houses) == 0:
            return jsonify({"error": "No hay casas generadas para ubicar hospitales"}), 400

        hospitals = find_hospital_positions(m, n, houses, k, max_iters=max_iters)
        hospitals_arr = np.array(hospitals)
        
        # Cálculo de estadísticas
        dists = np.sqrt(((houses[:, None, :] - hospitals_arr[None, :, :]) ** 2).sum(axis=2))
        min_dists = np.min(dists, axis=1)
        avg_d = float(np.mean(min_dists))
        max_d = float(np.max(min_dists))

        can_draw = (m <= 120 and n <= 120)
        image_b64 = None
        if can_draw:
            img_bytes = plot_map(m, n, houses, hospitals, title="Distribución Óptima de Hospitales")
            image_b64 = base64.b64encode(img_bytes).decode("utf-8")

        return jsonify({
            "hospitals": hospitals,
            "stats": {"avg_distance": avg_d, "max_distance": max_d},
            "can_draw": can_draw,
            "image_base64": image_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Importante para Docker y Render
    app.run(host="0.0.0.0", port=8000)