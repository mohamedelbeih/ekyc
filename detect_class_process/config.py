"""
    The config file holding the pydantic settings instance.
"""
import json
import os
import sys

from pydantic import BaseSettings


def read_config(path: str) -> dict:
    """
    This function reads the file containing the application config.

    Parameters
    ===========
        path: str
            The input file containing the error messages

    Returns
    ========
        (dict, bool)
            if bool is False, dict will contain an error key containing the error message like:
                {'error': f"Config file not found,...."}
            if bool is True, dic will contain a dict contain the config dict.

    """
    if not os.path.isfile(path):
        print(f"Config file not found, should be at {path}", flush=True)
        sys.exit(1)
    try:
        conf = json.load(open(path))
        return conf
    except (ValueError, KeyError):
        print(f"Couldn't parse config while reading file {path}", flush=True)
        sys.exit(1)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
CONFIG = read_config(CONFIG_PATH)


class Settings(BaseSettings):
    """
        This is a settings class which inherate from pydantic
        BaseSetting class and contain all system variable.
    """
    VERSION: str = CONFIG.get("VERSION")
    SERVER_PORT: int = CONFIG.get("SERVER_PORT")
    LAYOUT_THREADS: int = CONFIG.get("LAYOUT_THREADS")
    RECOGNITION_THREADS: int = CONFIG.get("RECOGNITION_THREADS")
    MODELS_DIR: str = CONFIG.get("MODELS_DIR")
    ALLOWED_CORS_ORIGINS: list = CONFIG.get("ALLOWED_CORS_ORIGINS")


settings = Settings()
