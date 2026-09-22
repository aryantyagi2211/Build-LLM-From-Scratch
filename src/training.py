import logging
import os
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


def train_model(model, dataset, val_dataset, epochs, learning_rate, checkpoint_dir, checkpoint_filename, initial_epoch):
    model = compile_model(model, learning_rate)

    if initial_epoch >= epochs:
        logger.warning(
            f"Checkpoint already at epoch {initial_epoch}, target epochs={epochs}. "
            f"Nothing to train — increase epochs in config to continue.")
        return None

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)



    nan_check_callback = tf.keras.callbacks.TerminateOnNaN()
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=False,
        monitor='val_loss',
        verbose=1
    )

    try:
        history = model.fit(
            dataset, 
            validation_data=val_dataset,
            epochs=epochs, 
            callbacks=[nan_check_callback, checkpoint_callback],
            initial_epoch=initial_epoch
            )
    except Exception as error:
        logger.error(f"Failed to train the model: {error}")
        raise
    
    for epoch_number, epoch_loss in enumerate(history.history['loss']):
        logger.info(f"Epoch {epoch_number}, Final Loss: {epoch_loss:.4f}")

    logger.info(f"Model trained for {epochs} epochs")
    return history
