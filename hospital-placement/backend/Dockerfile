FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el contenido necesario
COPY app.py .
COPY utils.py .
COPY placer.py .
# Si tienes algún CSV necesario, descomenta la siguiente línea:
# COPY data.csv .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Comando para Flask
CMD ["python", "app.py"]