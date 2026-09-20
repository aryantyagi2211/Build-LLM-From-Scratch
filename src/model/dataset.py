import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def create_batched_dataset(token_ids, batch_size, max_sequence_length):

    try:
        token_ids_tensor = tf.constant(token_ids)

        dataset = tf.data.Dataset.from_tensor_slices(token_ids)
        sequences = dataset.batch(max_sequence_length + 1, drop_remainder=True)

        def split_input_target(sequence):
            input_sequence = sequence[:-1]
            target_sequence = sequence[1:]
            return input_sequence, target_sequence

        dataset = sequences.map(split_input_target)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size, drop_remainder=True)
    except Exception as error:
        logger.error(f"Failed to create batched dataset: {error}")
        raise

    logger.info(f"Batched dataset created with batch size: {batch_size} and max sequence length: {max_sequence_length}")
    return dataset