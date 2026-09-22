import logging
import numpy as np

logger = logging.getLogger(__name__)

def calculate_perplexity(loss):
    perplexity = np.exp(loss)
    logger.info(f"Calculated perplexity: {perplexity}")
    return perplexity

def evaluate_model(model, val_dataset):
    results = model.evaluate(val_dataset)

    if isinstance(results, list):
        val_loss = results[0]
    else:
        val_loss = results
        
    perplexity = calculate_perplexity(val_loss)
    logger.info(f"Validation loss: {val_loss}, Perplexity: {perplexity:.4f}")

    return val_loss, perplexity