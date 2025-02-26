# 🔍 Computer Vision Service

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

A powerful and extensible service designed for computer vision tasks, including image classification, object detection, and specialized detection. Built with FastAPI and containerized using Docker, this service is a production-ready solution for deploying advanced computer vision models.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quickstart](#-quickstart)
- [Key Files](#-key-files)
- [Configuration](#-configuration)
- [API Endpoints](#-api-endpoints)
- [Rate Limiting](#-rate-limiting)
- [Extending the Service](#-extending-the-service)
  - [Adding New Models](#adding-new-models)
  - [Creating Custom Preprocessors](#creating-custom-preprocessors)
  - [Implementing Post-processors](#implementing-post-processors)
  - [Developing Prediction Strategies](#developing-prediction-strategies)
- [Testing](#-testing)
- [Docker Deployment](#-docker-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Image Classification**: Recognize objects in images using cutting-edge classification models like YOLO and EfficientNet.
- **Object Detection**: Detect and localize multiple objects within images, generating accurate bounding boxes.
- **Specialized Detection**: Use class-specific detection models for optimized results on specialized categories.
- **Extensible Architecture**: Seamlessly add new models, preprocessors, and prediction strategies via a simple registration system.
- **Rate Limiting**: Redis-based rate limiting to prevent service overuse and ensure fair access.
- **Dockerized Deployment**: Docker support for easy deployment, scaling, and management.
- **Comprehensive Testing**: Includes unit, integration, and API tests to ensure reliable operation.

## 🏗 Architecture

This service follows a modular architecture with the following core components:

```bash
src/
├── conf/ # Configuration files
├── schema/ # API schema definitions
├── service/
│ ├── constants/ # Constants for image processing
│ ├── models/ # Machine learning model implementations
│ │ ├── classification_models/
│ │ └── detection_models/
│ ├── pipelines/ # Processing pipelines for model flow
│ ├── prediction_strategies/ # Custom prediction strategies
│ ├── processing/ # Preprocessing and postprocessing components
│ ├── tools.py # Utility functions
│ └── service.py # FastAPI application entry point
├── settings/ # Service settings and configurations
└── logger.py # Logging configuration
```

### Key Components

- **Models**: Base model interface and specific implementations (YOLO, EfficientNet)
- **Pipelines**: Orchestrate the flow from preprocessing to prediction to postprocessing
- **Processing**: Transform inputs/outputs for model compatibility and user-friendly responses
- **Prediction Strategies**: Special handling for specific prediction scenarios (e.g., rotation)

## 🚀 Quickstart

### Docker Setup (Recommended)

The fastest way to get started with the CV service is using Docker:

```bash
# Clone the repository
git clone https://github.com/your-org/cv-service.git
cd cv-service
# Configure environment variables
cp .env.example .env
# Edit .env file to set your configuration
# Build and start the services
docker-compose up -d
# The service will be available at http://localhost:8000
# Access the API documentation at http://localhost:8000/docs
```

### Local Setup with Poetry

For local development:

```bash
# Clone the repository
git clone https://github.com/your-org/cv-service.git
cd cv-service
# Install dependencies
poetry install
# Configure environment variables
cp .env.example .env
# Edit .env file to set your configuration
# Activate the virtual environment
poetry shell
# Run the service
python src/run_service.py
# The service will be available at http://localhost:8000
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
