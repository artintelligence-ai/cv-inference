import logging
import os
import random
from os.path import join as opj
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    logger.info(f"🔄 Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logger.info("✅ Random seed set successfully")


def load_state_dict(
    bucket_name: str,
    path2weights: str,
    download_path: str,
    model_name: str,
    torch_ckpt: bool = False,
    logger=None,
) -> Any:
    """Load model state dictionary from local path or S3."""
    model_path = opj(download_path, model_name)

    if not os.path.exists(model_path):
        logger.warning(
            f"⚠️ Model file not found at {model_path}. Should load from S3: {path2weights}"
        )
        # TODO: Implement S3 download logic here

    if torch_ckpt:
        logger.info("🔄 Loading torch checkpoint")
        model_path = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=True
        )
        logger.info("✅ Torch checkpoint loaded successfully")
    return model_path


def model2cuda(device, model):
    """Move model to CUDA device if available."""
    if torch.cuda.is_available():
        logger.info(f"🔄 Moving model to device: {device}")
        model.to(torch.device(device))
        logger.info("✅ Model moved to CUDA successfully")
    return model
