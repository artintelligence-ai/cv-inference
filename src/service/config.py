import logging

from hydra import compose, initialize
from omegaconf import OmegaConf

from settings.pipelines import (
    SingleModelPipelineSettings,
    MultiModelPipelineSettings,
)

from settings.settings import settings

logger = logging.getLogger(__name__)


def hydra_args(config_name=settings.config_name):
    """
    Load configuration using Hydra.

    Args:
        config_name: Name of the config file (without extension)

    Returns:
        Loaded configuration object
    """
    logger.info(f"🔄 Loading configuration from {config_name}")
    try:
        with initialize(version_base=None, config_path="../conf"):
            cfg = compose(config_name=config_name)
            logger.debug(f"✅ Configuration loaded successfully")
            return cfg
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        raise


cfg = hydra_args()

try:
    classification_cfg = OmegaConf.to_container(
        cfg.get("image_classification", {}), resolve=True
    )
    classification_config = SingleModelPipelineSettings.parse_obj(classification_cfg)

    detection_cfg = OmegaConf.to_container(
        cfg.get("object_detection", {}), resolve=True
    )
    detection_config = SingleModelPipelineSettings.parse_obj(detection_cfg)

    classwise_cfg = OmegaConf.to_container(
        cfg.get("classwise_object_detection", {}), resolve=True
    )
    classwise_detection_config = MultiModelPipelineSettings.parse_obj(classwise_cfg)

    logger.info("✅ All configurations loaded and validated")
except Exception as e:
    logger.error(f"❌ Configuration validation error: {e}")
    raise
