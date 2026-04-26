#!/bin/bash

# ===========================================================
# Script de Instalación - Sistema KNN Predicción de Voto
# ===========================================================

echo "======================================"
echo "🗳️  Sistema de Predicción de Voto KNN"
echo "======================================"
echo ""

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Verificar si Docker está instalado
echo "Verificando requisitos..."
if ! command -v docker &> /dev/null; then
    print_error "Docker no está instalado"
    echo "Por favor instala Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
print_success "Docker está instalado"

# Verificar si Docker Compose está instalado
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose no está instalado"
    echo "Por favor instala Docker Compose"
    exit 1
fi
print_success "Docker Compose está instalado"

# Crear estructura de carpetas
echo ""
echo "Creando estructura de carpetas..."
mkdir -p backend frontend
print_success "Carpetas creadas"

# Verificar archivo CSV
echo ""
echo "Verificando dataset..."
if [ ! -f "voter_intentions_3000.csv" ]; then
    print_error "No se encontró el archivo voter_intentions_3000.csv"
    echo "Por favor coloca el archivo en este directorio"
    exit 1
fi
print_success "Dataset encontrado"

# Copiar CSV a backend
cp voter_intentions_3000.csv backend/
print_success "Dataset copiado a backend/"

# Crear archivo .env si no existe
if [ ! -f ".env" ]; then
    echo ""
    echo "Creando archivo .env..."
    cat > .env << EOF
# Configuración del Sistema
BACKEND_PORT=8000
FRONTEND_PORT=80
PYTHONUNBUFFERED=1
EOF
    print_success "Archivo .env creado"
fi

# Función para verificar si los archivos necesarios existen
check_files() {
    echo ""
    echo "Verificando archivos necesarios..."
    
    local missing_files=0
    
    if [ ! -f "backend/app.py" ]; then
        print_error "Falta: backend/app.py"
        missing_files=1
    fi
    
    if [ ! -f "backend/Dockerfile" ]; then
        print_error "Falta: backend/Dockerfile"
        missing_files=1
    fi
    
    if [ ! -f "backend/requirements.txt" ]; then
        print_error "Falta: backend/requirements.txt"
        missing_files=1
    fi
    
    if [ ! -f "frontend/index.html" ]; then
        print_error "Falta: frontend/index.html"
        missing_files=1
    fi
    
    if [ ! -f "frontend/nginx.conf" ]; then
        print_error "Falta: frontend/nginx.conf"
        missing_files=1
    fi
    
    if [ ! -f "docker-compose.yml" ]; then
        print_error "Falta: docker-compose.yml"
        missing_files=1
    fi
    
    if [ $missing_files -eq 1 ]; then
        echo ""
        print_error "Faltan archivos necesarios. Por favor asegúrate de tener todos los archivos."
        exit 1
    fi
    
    print_success "Todos los archivos están presentes"
}

# Verificar archivos
check_files

# Preguntar si desea construir ahora
echo ""
read -p "¿Deseas construir y ejecutar el sistema ahora? (s/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo ""
    echo "======================================"
    echo "Construyendo servicios..."
    echo "======================================"
    echo ""
    
    # Detener servicios existentes si los hay
    docker-compose down 2>/dev/null
    
    # Construir y levantar servicios
    docker-compose up --build -d
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "======================================"
        print_success "Sistema desplegado exitosamente!"
        echo "======================================"
        echo ""
        print_info "Esperando a que los servicios estén listos..."
        sleep 5
        
        # Verificar estado de los servicios
        echo ""
        echo "Estado de los servicios:"
        docker-compose ps
        
        echo ""
        echo "======================================"
        echo "🎉 ¡Sistema listo!"
        echo "======================================"
        echo ""
        echo "Accede a:"
        echo "  • Frontend: ${BLUE}http://localhost${NC}"
        echo "  • Backend API: ${BLUE}http://localhost:8000${NC}"
        echo "  • Documentación: ${BLUE}http://localhost:8000/docs${NC}"
        echo ""
        echo "Comandos útiles:"
        echo "  • Ver logs: ${BLUE}docker-compose logs -f${NC}"
        echo "  • Detener: ${BLUE}docker-compose down${NC}"
        echo "  • Reiniciar: ${BLUE}docker-compose restart${NC}"
        echo ""
        
        # Intentar abrir el navegador
        if command -v xdg-open &> /dev/null; then
            read -p "¿Deseas abrir el navegador? (s/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Ss]$ ]]; then
                xdg-open http://localhost 2>/dev/null
            fi
        elif command -v open &> /dev/null; then
            read -p "¿Deseas abrir el navegador? (s/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Ss]$ ]]; then
                open http://localhost 2>/dev/null
            fi
        fi
        
    else
        print_error "Error al desplegar el sistema"
        echo ""
        echo "Revisa los logs con: docker-compose logs"
        exit 1
    fi
else
    echo ""
    print_info "Puedes construir el sistema más tarde con:"
    echo "  ${BLUE}docker-compose up --build -d${NC}"
fi

echo ""
print_success "Instalación completada"