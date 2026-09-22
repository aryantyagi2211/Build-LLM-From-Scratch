import logging

logger = logging.getLogger(__name__)

def validate_input(prompt, max_prompt_length=500):
    if not prompt or not prompt.strip():
        logger.warning("Input prompt is empty or whitespace.")
        raise ValueError("Input prompt cannot be empty or whitespace.")

    if len(prompt) > max_prompt_length:
        logger.warning(f"Input prompt exceeds maximum length of {max_prompt_length}.")
        raise ValueError(f"Prompt too long, max {max_prompt_length} characters allowed")

    return True


def validate_output(generated_text):
    if not generated_text or not generated_text.strip():
        logger.warning("Generated text is empty or whitespace.")
        raise ValueError("Generated text cannot be empty or whitespace.")

    return True