import logging
import logging as log
import sys


def setup_logging(log_level="INFO", log_file="training.log"):
    log.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            log.FileHandler(log_file),
            log.StreamHandler()
        ]
    )