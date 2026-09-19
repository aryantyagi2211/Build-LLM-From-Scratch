import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def build_layer_norm():
    layer_norm_layer = tf.keras.layers.LayerNormalization()
    logger.info(f"Layer Normalization created.")
    return layer_norm_layer
