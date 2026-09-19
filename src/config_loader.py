import yaml
import logging

logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as error:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as error:
        logger.error(f"Invalid YAML syntax in {config_path}: {error}")
        raise

    return config