# Backend container for the finance-RAG API.
# Production uses VECTOR_STORE=supabase (the local NumPy index is dev-only and
# ephemeral in a container). Ingest into Supabase once, then this image serves.
FROM python:3.12-slim

WORKDIR /app

# install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# shell form so $PORT (set by Render/Fly/Railway) expands; falls back to 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
