import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def compile_model(model, learning_rate):
    try:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    except Exception as error:
        logger.error(f"Failed to compile the model: {error}")
        raise

    logger.info(f"Model compiled with learning rate: {learning_rate}")
    return model


def train_model(model, dataset, epochs, learning_rate):
    model = compile_model(model, learning_rate)
    nan_check_callback = tf.keras.callbacks.TerminateOnNaN()

    try:
        history = model.fit(dataset, epochs=epochs, callbacks=[nan_check_callback])
    except Exception as error:
        logger.error(f"Failed to train the model: {error}")
        raise

    logger.info(f"Model trained for {epochs} epochs")
    return history
