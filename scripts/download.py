from src.logger import setup_logging
from src.download_data import download_wikitext
from src.config_loader import load_config

config = load_config("config/config.yaml")
setup_logging(
    log_level=config["logging"]["log_level"],
    log_file=config["logging"]["log_file"],
)
download_wikitext(config["data"]["file_path"])