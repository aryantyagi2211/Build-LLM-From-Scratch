import tensorflow as tf
import logging
from src.config_loader import load_config
from src.logger import setup_logging
from src.data_handling.data_loader import load_txt_data, split_train_val
from src.cleaning.text_cleaner import clean_text
from src.tokenizer.tokenier_bpe import get_tokens, encode, decode
from src.model.embedding import build_embedding
from src.model.positional_encoding import build_positional_encoding
from src.model.transformer_block import build_transformer_stack, apply_transformer_stack
from src.model.output_layer import build_output_layer

config = load_config("config/config.yaml")
logger = logging.getLogger(__name__)

setup_logging(
    log_level=config["logging"]["log_level"],
    log_file=config["logging"]["log_file"]
)
raw_txt = load_txt_data(config["data"]["file_path"])


clean_txt = clean_text(raw_txt)


train_txt, val_txt = split_train_val(
    clean_txt,
    val_ratio=config["data"]["val_ratio"]
)

tokenizer = get_tokens(config["tokenizer"]["encoding_name"])
train_token_ids = encode(train_txt, tokenizer)
val_token_ids = encode(val_txt, tokenizer)

embedding_layer = build_embedding(
    vocab_size=config["model"]["vocab_size"],
    embedding_dim=config["model"]["embedding_dim"],
)

positional_encoding = build_positional_encoding(
    max_sequence_length=config["model"]["max_sequence_length"],
    embedding_dim=config["model"]["embedding_dim"],
)

max_sequence_length = config["model"]["max_sequence_length"]
sample_token_ids = tf.constant([train_token_ids[:max_sequence_length]])

token_embeddings = embedding_layer(sample_token_ids)

position = tf.range(max_sequence_length)
position_embeddings = positional_encoding(position)

final_embedding = token_embeddings + position_embeddings


transformer_blocks = build_transformer_stack(
    embedding_dim=config["model"]["embedding_dim"],
    num_heads=config["model"]["num_heads"],
    hidden_dim=config["model"]["hidden_dim"],
    activation=config["model"]["activation"],
    num_blocks=config["model"]["num_blocks"],
)

x = apply_transformer_stack(final_embedding, transformer_blocks)

output_layer = build_output_layer(vocab_size=config["model"]["vocab_size"])
logits = output_layer(x)