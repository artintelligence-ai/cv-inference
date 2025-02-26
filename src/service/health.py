import logging
import platform
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from settings.settings import settings

# Set up logger
logger = logging.getLogger(__name__)

# Create router
health_router = APIRouter(tags=["Health"])


# Health response model
class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    environment: str
    timestamp: str
    uptime: float
    system_info: dict[str, Any]


# Service start time for uptime calculation
START_TIME = time.time()


def get_system_info() -> dict[str, Any]:
    """Get system information for health check."""
    return {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


@health_router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint to verify service is running properly.

    Returns:
        HealthResponse: Health status information
    """
    logger.debug("🔍 Health check requested")
    uptime = time.time() - START_TIME

    return HealthResponse(
        status="healthy",
        environment=settings.environment,
        timestamp=datetime.now().isoformat(),
        uptime=uptime,
        system_info=get_system_info(),
    )
