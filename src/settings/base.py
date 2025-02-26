from pydantic import Extra
from pydantic_settings import BaseSettings


class ExtraFieldsNotAllowedBaseSettings(BaseSettings):
    class Config:
        extra = Extra.forbid
        protected_namespaces = ()
