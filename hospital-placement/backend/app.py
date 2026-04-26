from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
from utils import generate_houses, plot_map
from placer import find_hospital_positions

app = Flask(__name__)
CORS(app)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    try:
        m = int(data.get("m"))
        n = int(data.get("n"))
        num_houses = int(data.get("num_houses"))
        houses = generate_houses(m, n, num_houses).tolist()

        can_draw = (m <= 120 and n <= 120)
        return jsonify({"houses": houses, "can_draw": can_draw})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/place", methods=["POST"])
def place():
    data = request.get_json()
    try:
        m = int(data.get("m"))
        n = int(data.get("n"))
        houses = np.array(data.get("houses"))
        k = int(data.get("k"))
        max_iters = int(data.get("max_iters", 50))

        hospitals = find_hospital_positions(m, n, houses, k, max_iters=max_iters)
        hospitals_arr = np.array(hospitals)
        
        # Calcular distancias promedio y máxima
        if len(houses) > 0:
            dists = np.sqrt(((houses[:, None, :] - hospitals_arr[None, :, :]) ** 2).sum(axis=2))
            min_dists = np.min(dists, axis=1)
            avg_d = float(np.mean(min_dists))
            max_d = float(np.max(min_dists))
        else:
            avg_d = max_d = 0.0

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
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
