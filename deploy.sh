#!/bin/bash
set -e

echo "🚀 Building and starting CV service..."
docker compose up --build -d

echo "⏳ Waiting for service to start..."
sleep 5

echo "🔍 Checking service health..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Service is healthy!"
else
    echo "❌ Service health check failed!"
    docker-compose logs
fi 