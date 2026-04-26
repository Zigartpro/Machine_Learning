FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY voter_intentions_3000.csv .

ENV PYTHONUNBUFFERED=1

ENV PORT=8000

EXPOSE 8000

CMD uvicorn app:app --host 0.0.0.0 --port $PORT