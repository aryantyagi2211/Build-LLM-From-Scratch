import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def feed_forward(hidden_dim, embedding_dim, activation):
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_dim, activation=activation),
        tf.keras.layers.Dense(embedding_dim),
    ])
    logger.info(f'Feed Forward Layer created: embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, activation={activation}')
    return ffn