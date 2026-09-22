import glob
import logging
import os

import tensorflow as tf

logger = logging.getLogger(__name__)

def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.weights.h5"))

    if not checkpoint_files:
        logger.info(f"No checkpoint files found in {checkpoint_dir}")
        return None, 0

    try:
        latest_file = max(checkpoint_files, key=os.path.getctime) # getctime means file creation time
        logger.info(f"Latest checkpoint found: {latest_file}")
    except Exception as error:
        logger.error(f"Failed to find the latest checkpoint file in {checkpoint_dir}: {error}")
        raise

    epoch_num = int(os.path.basename(latest_file).split('_epoch_')[1].split('.')[0]) 
    
    return latest_file, epoch_num


def load_checkpoint_if_exists(model, checkpoint_dir, max_sequence_length):
    latest_file, last_epoch = find_latest_checkpoint(checkpoint_dir)

    if latest_file is None:
        logger.info("No checkpoint found. Starting training from scratch.")
        return model, 0

    dummy_input = tf.constant([[0] * max_sequence_length])
    model(dummy_input)

    try:
        model.load_weights(latest_file)
        logger.info(f"Loaded weights from {latest_file} for epoch {last_epoch}")
    except Exception as error:
        logger.error(f"Failed to load weights from {latest_file}: {error}")
        raise
    return model, last_epoch