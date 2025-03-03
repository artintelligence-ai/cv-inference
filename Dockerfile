FROM python:3.11.11-slim

WORKDIR /app

# Install system dependencies for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry with specific version
RUN pip install poetry==1.5.1

# Copy just pyproject.toml and poetry.lock first for better caching
COPY pyproject.toml poetry.lock* ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Copy application code
COPY configs/ ./conf
COPY src/schema/ ./schema
COPY src/service/ ./service
COPY src/settings/ ./settings
COPY src/__init__.py ./__init__.py
COPY src/logger.py ./logger.py
COPY src/run_service.py ./run_service.py

# Copy data
COPY data/ ./data/

# Install dependencies
RUN poetry install --only main --no-interaction --no-ansi

# Run the application
CMD ["python", "run_service.py"]

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1