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

# ════════════════════════════════════════════════════════════════
# INICIALIZACIÓN
# ════════════════════════════════════════════════════════════════

app = Flask(__name__)

# Configurar CORS
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',')
CORS(app, resources={r"/api/*": {"origins": [o.strip() for o in cors_origins]}})

print("🚀 Backend iniciando...")
print(f"CORS Origins: {cors_origins}")

# ════════════════════════════════════════════════════════════════
# FUNCIONES MATEMÁTICAS
# ════════════════════════════════════════════════════════════════

def distancia_euclidiana(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos"""
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
    """✅ Verifica que el backend esté funcionando"""
    print("📍 GET /api/health")
    return jsonify({
        'status': 'ok',
        'message': 'Backend K-Means funcionando correctamente',
        'version': '1.0.0'
    }), 200

@app.route('/api/generar-casas', methods=['POST'])
def generar_casas():
    """
    ✅ Genera casas aleatorias
    
    Body (JSON):
    {
        "num_casas": 40,
        "grid_x": 100,
        "grid_y": 100
    }
    """
    try:
        print("📍 POST /api/generar-casas")
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Body vacío'}), 400
        
        num_casas = data.get('num_casas', 40)
        grid_x = data.get('grid_x', 100)
        grid_y = data.get('grid_y', 100)
        
        print(f"   - Generando {num_casas} casas en grid {grid_x}x{grid_y}")
        
        casas = []
        for i in range(num_casas):
            casas.append({
                'id': i,
                'x': random.uniform(10, grid_x - 10),
                'y': random.uniform(10, grid_y - 10),
                'tipo': 'casa'
            })
        
        print(f"   ✅ {len(casas)} casas generadas")
        
        return jsonify({
            'success': True,
            'casas': casas,
            'total': len(casas)
        }), 200
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/inicializar-hospitales', methods=['POST'])
def inicializar_hospitales():
    """
    ✅ Inicializa hospitales de forma aleatoria desde casas
    
    Body (JSON):
    {
        "num_hospitales": 3,
        "casas": [...]
    }
    """
    try:
        print("📍 POST /api/inicializar-hospitales")
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Body vacío'}), 400
        
        num_hospitales = data.get('num_hospitales', 3)
        casas = data.get('casas', [])
        
        if not casas:
            return jsonify({'error': 'No hay casas'}), 400
        
        print(f"   - Inicializando {num_hospitales} hospitales")
        
        # Seleccionar índices aleatorios
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
        
        print(f"   ✅ {len(hospitales)} hospitales inicializados")
        
        return jsonify({
            'success': True,
            'hospitales': hospitales,
            'total': len(hospitales)
        }), 200
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/iteracion-kmeans', methods=['POST'])
def iteracion_kmeans():
    """
    ✅ Realiza una iteración del K-means
    
    Body (JSON):
    {
        "casas": [...],
        "hospitales": [...]
    }
    """
    try:
        print("📍 POST /api/iteracion-kmeans")
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Body vacío'}), 400
        
        casas = data.get('casas', [])
        hospitales = data.get('hospitales', [])
        
        if not casas or not hospitales:
            return jsonify({'error': 'Faltan casas u hospitales'}), 400
        
        print(f"   - Procesando {len(casas)} casas y {len(hospitales)} hospitales")
        
        asignaciones, nuevos_hospitales, movimiento_total = \
            realizar_iteracion_kmeans(casas, hospitales)
        
        convergencia = movimiento_total < 0.5
        
        print(f"   - Movimiento: {movimiento_total:.4f}")
        print(f"   - Convergencia: {convergencia}")
        
        return jsonify({
            'success': True,
            'asignaciones': asignaciones,
            'hospitales': nuevos_hospitales,
            'movimiento_total': movimiento_total,
            'convergencia': convergencia
        }), 200
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/multiples-iteraciones', methods=['POST'])
def multiples_iteraciones():
    """
    ✅ Realiza múltiples iteraciones automáticamente
    
    Body (JSON):
    {
        "casas": [...],
        "hospitales": [...],
        "num_iteraciones": 20
    }
    """
    try:
        print("📍 POST /api/multiples-iteraciones")
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Body vacío'}), 400
        
        casas = data.get('casas', [])
        hospitales = data.get('hospitales', [])
        num_iteraciones = data.get('num_iteraciones', 20)
        
        if not casas or not hospitales:
            return jsonify({'error': 'Faltan casas u hospitales'}), 400
        
        print(f"   - Ejecutando {num_iteraciones} iteraciones")
        
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
            
            print(f"   - Iteración {i+1}: movimiento={movimiento:.4f}")
            
            if movimiento < 0.5:
                print(f"   ✅ Convergencia alcanzada en iteración {i+1}")
                break
        
        historico['hospitales_final'] = hospitales
        historico['asignaciones_final'] = asignaciones
        
        return jsonify({
            'success': True,
            'data': historico,
            'iteraciones_realizadas': len(historico['iteraciones']),
            'convergencia_alcanzada': movimiento < 0.5
        }), 200
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/estadisticas', methods=['POST'])
def calcular_estadisticas():
    """
    ✅ Calcula estadísticas del resultado final
    
    Body (JSON):
    {
        "casas": [...],
        "hospitales": [...],
        "asignaciones": [...]
    }
    """
    try:
        print("📍 POST /api/estadisticas")
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Body vacío'}), 400
        
        casas = data.get('casas', [])
        hospitales = data.get('hospitales', [])
        asignaciones = data.get('asignaciones', [])
        
        if len(asignaciones) != len(casas):
            return jsonify({'error': 'Asignaciones no coinciden'}), 400
        
        print(f"   - Calculando estadísticas para {len(hospitales)} hospitales")
        
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
        
        print(f"   ✅ Estadísticas calculadas")
        
        return jsonify({
            'success': True,
            'estadisticas': estadisticas
        }), 200
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

# ════════════════════════════════════════════════════════════════
# MANEJO DE ERRORES
# ════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(error):
    """Endpoint no encontrado"""
    print(f"❌ 404 - Endpoint no encontrado")
    return jsonify({
        'error': 'Endpoint no encontrado',
        'mensaje': 'Verifica la URL de la petición'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Método no permitido"""
    print(f"❌ 405 - Método no permitido")
    return jsonify({
        'error': 'Método no permitido',
        'mensaje': 'Usa GET o POST según corresponda'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Error interno del servidor"""
    print(f"❌ 500 - Error interno")
    return jsonify({
        'error': 'Error interno del servidor',
        'mensaje': str(error)
    }), 500

# ════════════════════════════════════════════════════════════════
# RUTA DE PRUEBA
# ════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def index():
    """Página de bienvenida"""
    return jsonify({
        'mensaje': '🏥 API K-Means 3D - Backend',
        'version': '1.0.0',
        'endpoints': {
            'GET /api/health': 'Verificar estado',
            'POST /api/generar-casas': 'Generar casas aleatorias',
            'POST /api/inicializar-hospitales': 'Inicializar hospitales',
            'POST /api/iteracion-kmeans': 'Una iteración',
            'POST /api/multiples-iteraciones': 'Múltiples iteraciones',
            'POST /api/estadisticas': 'Calcular estadísticas'
        }
    }), 200

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print("\n" + "="*60)
    print("🚀 K-MEANS 3D BACKEND INICIADO")
    print("="*60)
    print(f"📍 Host: {host}")
    print(f"📍 Puerto: {port}")
    print(f"📍 Debug: {debug}")
    print(f"📍 URL: http://{host}:{port}")
    print("\n📚 Endpoints disponibles:")
    print("   - GET  http://localhost:5000/")
    print("   - GET  http://localhost:5000/api/health")
    print("   - POST http://localhost:5000/api/generar-casas")
    print("   - POST http://localhost:5000/api/inicializar-hospitales")
    print("   - POST http://localhost:5000/api/iteracion-kmeans")
    print("   - POST http://localhost:5000/api/multiples-iteraciones")
    print("   - POST http://localhost:5000/api/estadisticas")
    print("="*60 + "\n")
    
    app.run(host=host, port=port, debug=debug)