import tensorflow as tf
import logging
from src.config_loader import load_config
from src.logger import setup_logging
from src.data_handling.data_loader import load_txt_data, split_train_val
from src.cleaning.text_cleaner import clean_text
from src.tokenizer.tokenier_bpe import get_tokens, encode, decode
from src.model.embedding import build_embedding
from src.model.positional_encoding import build_positional_encoding
from src.model.attention import build_attention_model
from src.model.feed_forward import feed_forward
from src.model.layer_norm import build_layer_norm          # <-- naya import

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

attention_layer = build_attention_model(
    embedding_dim=config["model"]["embedding_dim"],
    num_heads=config["model"]["num_heads"],

)
logger.info(type(attention_layer))

feed_forward_layer = feed_forward(
    embedding_dim=config["model"]["embedding_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    activation=config["model"]["activation"],
)

layer_norm_1 = build_layer_norm()          # <-- naya, attention se pehle
layer_norm_2 = build_layer_norm()          # <-- naya, feed forward se pehle

x = final_embedding                                     # <-- naya, starting point

normalized_x = layer_norm_1(x)                           # <-- naya
attention_outputs = attention_layer(
    query=normalized_x,
    key=normalized_x,
    value=normalized_x,
    use_causal_mask=True,
)
x = x + attention_outputs                                # <-- naya, residual #1

normalized_x = layer_norm_2(x)                            # <-- naya
feed_forward_output = feed_forward_layer(normalized_x)
x = x + feed_forward_output                               # <-- naya, residual #2