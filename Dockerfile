FROM python:3.12-slim

WORKDIR /app

COPY api/requirements.txt .

# Install xgboost without its GPU-related dependencies
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 xgboost==3.4.1 --no-deps

# Install everything else normally
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port $PORT"]