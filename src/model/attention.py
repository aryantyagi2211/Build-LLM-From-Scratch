import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def build_attention_model(num_heads, embedding_dim):
    try:
        attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim
        )
    except Exception as error:
        logger.error(f'Attention layer is not created {error}')
        raise

    logger.info(f'Attention layer created: num heads={num_heads}, embedding_dim={embedding_dim}')
    return attention_layer
