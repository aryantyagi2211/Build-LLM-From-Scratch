import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def build_positional_encoding(max_sequence_length, embedding_dim):
    position_encoding = tf.keras.layers.Embedding(
        max_sequence_length,
        embedding_dim
    )
    logger.info(f"Created positional encoding layer: max_sequence_length={max_sequence_length}, embedding_dim={embedding_dim}")
    return position_encoding