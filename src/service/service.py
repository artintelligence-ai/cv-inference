import logging
import time
from concurrent.futures.process import ProcessPoolExecutor
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.exceptions import HTTPException

from schema.schema import ClassificationOutput, DetectionOutput
from service.config import (
    classification_config,
    classwise_detection_config,
    detection_config,
)
from service.health import health_router
from service.pipelines.base import BasePipeline
from service.tools import (
    load_and_process_image,
    run_image_classification,
    run_object_detection,
)
from service.utils import set_random_seed
from settings.settings import settings

# Set a fixed seed for reproducibility
set_random_seed(seed=17)

# Create the logger - but don't set up logging itself (handled in run_service.py)
logger = logging.getLogger("CVService")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources for the application during startup and cleanup on shutdown."""
    try:
        # Store logger in app state for potential use in middleware or endpoints
        app.state.logger = logger
        logger.info("🚀 Initializing Computer Vision Service...")

        # Initialize the pipelines
        logger.info("🔄 Loading classification pipeline...")
        app.state.classification_pipeline = BasePipeline.by_name(
            classification_config.pipeline_class
        )(
            config=classification_config,
            device=settings.device,
            local_weights=settings.local_weights,
        )

        logger.info("🔄 Loading detection pipeline...")
        app.state.detection_pipeline = BasePipeline.by_name(
            detection_config.pipeline_class
        )(
            config=detection_config,
            device=settings.device,
            local_weights=settings.local_weights,
        )

        logger.info("🔄 Loading classwise detection pipeline...")
        app.state.classwise_detection_pipeline = BasePipeline.by_name(
            classwise_detection_config.pipeline_class
        )(
            config=classwise_detection_config,
            device=settings.device,
            local_weights=settings.local_weights,
        )

        # Store available class-specific models for reference
        if hasattr(app.state.classwise_detection_pipeline, "models"):
            app.state.available_detection_models = list(
                app.state.classwise_detection_pipeline.models.keys()
            )
            logger.info(
                f"ℹ️ Available class-specific detection models: {app.state.available_detection_models}"
            )

        logger.info("✅ All pipelines initialized successfully")

        # Initialize process executor for parallel processing
        app.state.executor = ProcessPoolExecutor(max_workers=2)
        logger.info("✅ Process executor initialized")

        yield

        # Cleanup resources
        app.state.executor.shutdown(wait=True)
        logger.info("🛑 Application shutdown and resources released.")

    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        # Still yield to allow the application to start even with errors
        yield
        raise


# Create the FastAPI app and router
app = FastAPI(
    title="Computer Vision Service",
    description="API for computer vision tasks like classification and detection",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Include health check router
app.include_router(health_router)

# Create main router
router = APIRouter(tags=["Vision"])

# Register main router with app
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    # TODO: Redis and rate limiting code
    pass


@app.post(
    "/api/v1/classify",
    tags=["Vision"],
    response_model=ClassificationOutput,
)
async def classify_image_endpoint(image: UploadFile = File(...)):
    """
    Endpoint for image classification.

    Args:
        image: The image file to be classified

    Returns:
        ClassificationOutput with classification results
    """
    logger.info(f"🔄 Received request to classify image: {image.filename or 'unknown'}")

    # Process the image
    processed_image = await load_and_process_image(image, logger)

    # Run classification
    return await run_image_classification(
        classification_pipeline=app.state.classification_pipeline,
        image=processed_image,
        logger=logger,
    )


@app.post(
    "/api/v1/detect",
    tags=["Vision"],
    response_model=DetectionOutput,
)
async def detect_objects_endpoint(image: UploadFile = File(...)):
    """
    Endpoint for general object detection.

    Args:
        image: The image file for object detection

    Returns:
        DetectionOutput with detected objects
    """
    logger.info(
        f"🔄 Received request for object detection: {image.filename or 'unknown'}"
    )

    # Process the image
    processed_image = await load_and_process_image(image, logger)

    # Run detection
    return await run_object_detection(
        detection_pipeline=app.state.detection_pipeline,
        image=processed_image,
        logger=logger,
    )


@app.post(
    "/api/v1/classwise-detect",
    tags=["Vision"],
    response_model=DetectionOutput,
)
async def classwise_detect_objects_endpoint(
    image: UploadFile = File(...),
    primary_class: str = None,
):
    """
    Endpoint for class-specific object detection.
    Uses specialized detection models for specific classes when available.

    Args:
        image: The image file for object detection
        primary_class: The specific class to use a specialized detection model for

    Returns:
        DetectionOutput with detected objects

    Raises:
        HTTPException: If primary_class is not provided or no model exists for it
    """
    logger.info(
        f"🔄 Received request for class-specific detection. "
        f"Image: {image.filename or 'unknown'}, Primary class: {primary_class}"
    )

    if not primary_class:
        logger.error("❌ Missing required parameter: primary_class")
        raise HTTPException(
            status_code=400,
            detail="primary_class parameter is required for class-specific detection",
        )

    if (
        not hasattr(app.state, "available_detection_models")
        or primary_class not in app.state.available_detection_models
    ):
        logger.error(f"❌ No specialized model available for class: {primary_class}")
        raise HTTPException(
            status_code=400,
            detail=f"No specialized detection model available for class: {primary_class}",
        )

    # Process the image
    processed_image = await load_and_process_image(image, logger)

    # Run class-specific detection
    return await run_object_detection(
        detection_pipeline=app.state.classwise_detection_pipeline,
        image=processed_image,
        logger=logger,
        primary_class=primary_class,
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add processing time to response headers."""
    start_time = time.time()
    logger.debug(f"Processing request to: {request.url.path}")
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.debug(f"Request processed in {process_time:.3f} seconds: {request.url.path}")
    return response


@app.get("/api/v1/test", tags=["Vision"])
async def test_endpoint():
    """Simple test endpoint to verify API routing."""
    return {"status": "ok", "message": "API router is working"}


# Add a simple test endpoint directly to the app (not the router)
@app.get("/direct-test", tags=["Test"])
async def direct_test():
    """Test endpoint directly on the app."""
    return {"status": "ok", "message": "Direct test endpoint is working"}
