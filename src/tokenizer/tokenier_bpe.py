import logging
import tiktoken
from src.config_loader import load_config

config = load_config("config/config.yaml")

logger = logging.getLogger(__name__)

def get_tokens(encoding_name):
    tokenizer = tiktoken.get_encoding(encoding_name)
    return tokenizer

def encode(text, tokenizer):
    tokens_ids = tokenizer.encode(text)
    logger.info(f"tokens_ids: {len(text)} characters info {len(tokens_ids)} tokens")
    return tokens_ids

def decode(tokens_ids, tokenizer):
    text = tokenizer.decode(tokens_ids)
    return text