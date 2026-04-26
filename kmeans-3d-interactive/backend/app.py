"""
════════════════════════════════════════════════════════════════
BACKEND - K-MEANS 3D INTERACTIVO
API REST con Flask
════════════════════════════════════════════════════════════════
"""

import os
import math
import random
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configurar CORS
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, resources={r"/api/*": {"origins": cors_origins}})

# ════════════════════════════════════════════════════════════════
# FUNCIONES MATEMÁTICAS
# ════════════════════════════════════════════════════════════════

def distancia_euclidiana(p1, p2):
    """Calcula distancia euclidiana entre dos puntos"""
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def centroide(puntos):
    """Calcula el centroide de un conjunto de puntos"""
    if not puntos:
        return None
    x_sum = sum(p['x'] for p in puntos)
    y_sum = sum(p['y'] for p in puntos)
    return {
        'x': x_sum / len(puntos),
        'y': y_sum / len(puntos)
    }

def realizar_iteracion_kmeans(casas, hospitales):
    """
    Realiza una iteración del algoritmo K-means
    
    Retorna:
    - asignaciones: lista de índices del hospital más cercano
    - nuevos_hospitales: posiciones actualizadas
    - movimiento_total: suma de movimientos
    """
    # PASO 1: Asignación
    asignaciones = []
    for casa in casas:
        distancias = [distancia_euclidiana(casa, h) for h in hospitales]
        hospital_cercano = distancias.index(min(distancias))
        asignaciones.append(hospital_cercano)
    
    # PASO 2: Recalcular centroides
    nuevos_hospitales = []
    movimiento_total = 0
    
    for h_idx in range(len(hospitales)):
        casas_grupo = [casas[i] for i in range(len(casas)) 
                      if asignaciones[i] == h_idx]
        
        if casas_grupo:
            nuevo_centro = centroide(casas_grupo)
            movimiento = distancia_euclidiana(hospitales[h_idx], nuevo_centro)
            movimiento_total += movimiento
            nuevos_hospitales.append(nuevo_centro)
        else:
            nuevos_hospitales.append(hospitales[h_idx])
    
    return asignaciones, nuevos_hospitales, movimiento_total

# ════════════════════════════════════════════════════════════════
# ENDPOINTS DE LA API
# ════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verifica que el backend esté funcionando"""
    return jsonify({
        'status': 'ok',
        'message': 'Backend K-Means funcionando correctamente',
        'version': '1.0.0'
    })

@app.route('/api/generar-casas', methods=['POST'])
def generar_casas():
    """
    Genera casas aleatorias
    
    Body:
    {
        "num_casas": 40,
        "grid_x": 100,
        "grid_y": 100
    }
    """
    try:
        data = request.json
        num_casas = data.get('num_casas', 40)
        grid_x = data.get('grid_x', 100)
        grid_y = data.get('grid_y', 100)
        
        casas = []
        for i in range(num_casas):
            casas.append({
                'id': i,
                'x': random.uniform(10, grid_x - 10),
                'y': random.uniform(10, grid_y - 10),
                'tipo': 'casa'
            })
        
        return jsonify({
            'success': True,
            'casas': casas,
            'total': len(casas)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/inicializar-hospitales', methods=['POST'])
def inicializar_hospitales():
    """
    Inicializa hospitales de forma aleatoria desde casas
    
    Body:
    {
        "num_hospitales": 3,
        "casas": [...]
    }
    """
    try:
        data = request.json
        num_hospitales = data.get('num_hospitales', 3)
        casas = data.get('casas', [])
        
        if not casas:
            return jsonify({'error': 'No hay casas'}), 400
        
        indices = random.sample(range(len(casas)), 
                               min(num_hospitales, len(casas)))
        
        hospitales = []
        for idx, i in enumerate(indices):
            hospitales.append({
                'id': idx,
                'x': casas[i]['x'],
                'y': casas[i]['y'],
                'tipo': 'hospital',
                'historico': [[casas[i]['x'], casas[i]['y']]]
            })
        
        return jsonify({
            'success': True,
            'hospitales': hospitales,
            'total': len(hospitales)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/iteracion-kmeans', methods=['POST'])
def iteracion_kmeans():
    """
    Realiza una iteración del K-means
    
    Body:
    {
        "casas": [...],
        "hospitales": [...]
    }
    """
    try:
        data = request.json
        casas = data.get('casas', [])
        hospitales = data.get('hospitales', [])
        
        if not casas or not hospitales:
            return jsonify({'error': 'Faltan casas u hospitales'}), 400
        
        asignaciones, nuevos_hospitales, movimiento = \
            realizar_iteracion_kmeans(casas, hospitales)
        
        return jsonify({
            'success': True,
            'asignaciones': asignaciones,
            'hospitales': nuevos_hospitales,
            'movimiento_total': movimiento,
            'convergencia': movimiento < 0.5
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/multiples-iteraciones', methods=['POST'])
def multiples_iteraciones():
    """
    Realiza múltiples iteraciones automáticamente
    
    Body:
    {
        "casas": [...],
        "hospitales": [...],
        "num_iteraciones": 20
    }
    """
    try:
        data = request.json
        casas = data.get('casas', [])
        hospitales = data.get('hospitales', [])
        num_iteraciones = data.get('num_iteraciones', 20)
        
        if not casas or not hospitales:
            return jsonify({'error': 'Faltan casas u hospitales'}), 400
        
        historico = {
            'iteraciones': [],
            'hospitales_final': hospitales,
            'asignaciones_final': []
        }
        
        for i in range(num_iteraciones):
            asignaciones, hospitales, movimiento = \
                realizar_iteracion_kmeans(casas, hospitales)
            
            historico['iteraciones'].append({
                'numero': i + 1,
                'hospitales': hospitales,
                'asignaciones': asignaciones,
                'movimiento': movimiento
            })
            
            if movimiento < 0.5:
                break
        
        historico['hospitales_final'] = hospitales
        historico['asignaciones_final'] = asignaciones
        
        return jsonify({
            'success': True,
            'data': historico,
            'iteraciones_realizadas': len(historico['iteraciones']),
            'convergencia_alcanzada': movimiento < 0.5
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/estadisticas', methods=['POST'])
def calcular_estadisticas():
    """
    Calcula estadísticas del resultado final
    
    Body:
    {
        "casas": [...],
        "hospitales": [...],
        "asignaciones": [...]
    }
    """
    try:
        data = request.json
        casas = data.get('casas', [])
        hospitales = data.get('hospitales', [])
        asignaciones = data.get('asignaciones', [])
        
        if len(asignaciones) != len(casas):
            return jsonify({'error': 'Asignaciones no coinciden'}), 400
        
        estadisticas = {
            'casas_por_hospital': [],
            'distancia_promedio': [],
            'distancia_total': 0
        }
        
        for h_idx in range(len(hospitales)):
            casas_grupo = [casas[i] for i in range(len(casas)) 
                          if asignaciones[i] == h_idx]
            
            distancias = [distancia_euclidiana(casa, hospitales[h_idx]) 
                         for casa in casas_grupo]
            
            estadisticas['casas_por_hospital'].append(len(casas_grupo))
            
            if distancias:
                prom = sum(distancias) / len(distancias)
                estadisticas['distancia_promedio'].append(prom)
                estadisticas['distancia_total'] += sum(distancias)
            else:
                estadisticas['distancia_promedio'].append(0)
        
        return jsonify({
            'success': True,
            'estadisticas': estadisticas
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ════════════════════════════════════════════════════════════════
# MANEJO DE ERRORES
# ════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', False)
    
    print(f"🚀 Servidor iniciado en http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)