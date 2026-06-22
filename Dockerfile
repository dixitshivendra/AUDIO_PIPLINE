FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg libpq-dev gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Add the current directory to Python path
ENV PYTHONPATH=/app
