from flask import Flask, request, jsonify
from flask_cors import CORS
from knn_model import MusicRecommenderSystem
import os

app = Flask(__name__)
CORS(app)

# Inicializar el sistema de recomendación
CSV_PATH = os.getenv('CSV_PATH', 'dataset_canciones_colombia_knn.csv')
recommender = None

try:
    recommender = MusicRecommenderSystem(CSV_PATH, k=5, metrica="coseno")
    print("Sistema de recomendación inicializado correctamente")
except Exception as e:
    print(f"Error al inicializar el sistema: {e}")

@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint de salud"""
    return jsonify({"status": "ok", "message": "Backend funcionando correctamente"})

@app.route('/api/canciones', methods=['GET'])
def obtener_canciones():
    """Retorna todas las canciones organizadas por género"""
    try:
        canciones_por_genero = recommender.obtener_canciones_por_genero()
        return jsonify(canciones_por_genero)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recomendar', methods=['POST'])
def recomendar():
    """
    Recibe ratings del usuario y retorna recomendaciones
    Body esperado:
    {
        "ratings": {
            "La Piragua": 5,
            "Mi Gente": 4,
            ...
        },
        "top_n": 10
    }
    """
    try:
        data = request.json
        ratings = data.get('ratings', {})
        top_n = data.get('top_n', 10)
        
        if not ratings:
            return jsonify({"error": "Debes proporcionar al menos una canción calificada"}), 400
        
        resultado = recommender.predecir_y_recomendar(ratings, top_n)
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generos', methods=['GET'])
def obtener_generos():
    """Retorna la lista de géneros disponibles"""
    try:
        generos = list(recommender.obtener_canciones_por_genero().keys())
        return jsonify({"generos": generos})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)