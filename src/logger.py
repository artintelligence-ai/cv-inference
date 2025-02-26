import functools
import logging
import os
import time


class CustomFormatter(logging.Formatter):
    """Custom log formatter to enhance log readability."""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging():
    """Set up logging configuration."""
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    root = logging.getLogger()

    # Set log level based on environment
    env = os.getenv("ENVIRONMENT", "dev").lower()
    if env == "prod":
        root.setLevel(logging.INFO)
    else:
        root.setLevel(logging.DEBUG)

    # Filter out logs from external libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root.addHandler(handler)


def log_execution_time(func):
    """Decorator to log the execution time of an async function."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)  # Await the async function
        end_time = time.time()
        logger = logging.getLogger(func.__module__)
        logger.info(
            f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds"
        )
        return result

    return async_wrapper
