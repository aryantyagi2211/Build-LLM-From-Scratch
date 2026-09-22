import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


def generate_text(model, tokenizer, prompt, max_new_tokens, max_sequence_length):
    try:
        token_ids = tokenizer.encode(prompt)
    except Exception as error:
        logger.error(f"Failed to encode the prompt: {error}")
        raise

    try:
        for _ in range(max_new_tokens):
            input_ids = token_ids[-max_sequence_length:]
            input_tensor = tf.constant([input_ids])

            logits = model(input_tensor)
            last_token_logits = logits[:, -1, :]

            next_token_id = int(tf.argmax(last_token_logits[0]))
            token_ids.append(next_token_id)

        generated_text = tokenizer.decode(token_ids)
    except Exception as error:
        logger.error(f"Failed to generate text: {error}")
        raise

    return generated_text