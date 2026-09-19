import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def build_embedding(vocab_size, embedding_dim):
    try:
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
        )
    except Exception as error:
        logger.error(f"Embedding Failed to created {error}")
        raise

    logger.info(f"Embedding layer created: Vocab size: {vocab_size} and embedding dim: {embedding_dim}")
    return embedding_layer