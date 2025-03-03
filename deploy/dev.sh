#!/bin/bash
set -e

echo "🚀 Starting CV service in development mode..."
docker compose watch

# This will keep running until you press Ctrl+C 