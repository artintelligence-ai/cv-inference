from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Redis settings
    redis_url: str = "redis://redis:6379/0"

    # Device settings
    device: str = "cpu"
    local_weights: str = "data/weights"

    # Config settings
    config_name: str = "config"

    # Environment settings
    environment: str = "dev"
    log_level: str = "INFO"

    # AWS credentials
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"
    aws_endpoint_url: str = "https://s3.amazonaws.com"

    # Service URLs
    ml_inference_url: str = "http://cv_qualifier:8002"

    # Rate limiting settings
    rate_limit_times: int = 10
    rate_limit_seconds: int = 60

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_prefix": "",
    }


settings = Settings()
