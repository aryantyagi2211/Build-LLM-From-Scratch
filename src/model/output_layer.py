import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


def build_output_layer(vocab_size):
    output_layer = tf.keras.layers.Dense(vocab_size)
    logger.info(f"Output layer created: vocab_size={vocab_size}")
    return output_layer