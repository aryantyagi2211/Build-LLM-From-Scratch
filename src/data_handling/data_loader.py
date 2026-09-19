from src.data_handling.validation import validate_data_file
import logging

logger = logging.getLogger(__name__)


def load_txt_data(filepath):
    validate_data_file(filepath)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_txt = f.read()
    except (IOError, OSError, UnicodeDecodeError) as error:
        logger.error(f"Failed to read file {filepath}: {error}")
        raise

    logger.info(f"Loaded {len(raw_txt)} characters from {filepath}")
    return raw_txt


def split_train_val(raw_txt, val_ratio=0.1):
    split_index =int(len(raw_txt) * (1.0 - val_ratio))
    train_txt = raw_txt[:split_index]
    val_txt = raw_txt[split_index:]

    logger.info(f"Train size: {len(train_txt)} chars, val_size: {len(val_txt)} chars")
    return train_txt, val_txt




