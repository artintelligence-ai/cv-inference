import logging

import uvicorn
from dotenv import load_dotenv

from logger import setup_logging
from settings.settings import settings

# Load environment variables
load_dotenv()

# Set up custom logging configuration
setup_logging()

logger = logging.getLogger(__name__)


def start_service():
    """Start the FastAPI service with the specified settings."""
    # Set up logging
    setup_logging()

    # Log startup information
    logger.info(
        f"🚀 Starting Computer Vision Inference Service on {settings.host}:{settings.port}"
    )
    logger.info(f"🔄 Development mode (hot reload): {settings.environment == 'dev'}")

    # Start the service
    uvicorn.run(
        "service.service:app",
        host=settings.host,
        port=settings.port,
        log_level="debug",
        reload=settings.environment == "dev",
        reload_dirs=["src"] if settings.environment == "dev" else None,
    )


if __name__ == "__main__":
    # Start the service with default settings
    start_service()
