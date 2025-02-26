import logging
from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image

from logger import log_execution_time
from schema.schema import ClassificationOutput, DetectionOutput

logger = logging.getLogger(__name__)


@log_execution_time
async def load_and_process_image(image_file: UploadFile, logger) -> Image.Image:
    """
    Utility function to load and preprocess an image.

    Args:
        image_file: The uploaded image file

    Returns:
        Processed PIL Image

    Raises:
        HTTPException: If image loading fails
    """
    try:
        logger.info("🔄 Loading and processing image")
        contents = await image_file.read()
        image = Image.open(BytesIO(contents))
        image = image.convert("RGB")
        logger.info("✅ Image loaded and processed successfully")
        return image
    except Exception as e:
        logger.error(f"❌ Failed to load image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


@log_execution_time
async def run_image_classification(
    classification_pipeline, image: Image.Image, logger, **prediction_kwargs
) -> ClassificationOutput:
    """Run classification on the provided image."""
    logger.info("🟢 Running classification model")
    try:
        prediction: ClassificationOutput = await classification_pipeline.predict(
            image=image, **prediction_kwargs
        )
        logger.info(
            f"🟢 Classification result: {prediction.label} ({prediction.probability:.3f})"
        )
        return prediction
    except Exception as e:
        logger.error(f"🔴 Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@log_execution_time
async def run_object_detection(
    detection_pipeline, image: Image.Image, logger, **prediction_kwargs
) -> DetectionOutput:
    """Run object detection on the provided image."""
    try:
        logger.info("🔄 Running detection model")
        prediction: DetectionOutput = await detection_pipeline.predict(
            image=image, **prediction_kwargs
        )

        total_objects = sum(cls_obj.count for cls_obj in prediction.root.values())
        detected_classes = list(prediction.root.keys())
        logger.info(
            f"✅ Detected {total_objects} objects of classes: {detected_classes}"
        )
        return prediction
    except Exception as e:
        logger.error(f"❌ Detection error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Object detection failed: {str(e)}"
        )
