import logging
import tensorflow as tf

from src.model.embedding import build_embedding
from src.model.positional_encoding import build_positional_encoding
from src.model.transformer_block import build_transformer_stack, apply_transformer_stack
from src.model.output_layer import build_output_layer

logger = logging.getLogger(__name__)

class GPTModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_blocks, max_sequence_length, activation):
        super().__init__()
        self.embedding_layer = build_embedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
        self.positional_encoding = build_positional_encoding(max_sequence_length=max_sequence_length, embedding_dim=embedding_dim)
        self.transformer_blocks = build_transformer_stack(embedding_dim, num_heads, hidden_dim, activation, num_blocks)
        self.output_layer = build_output_layer(vocab_size=vocab_size)

    def call(self, token_ids):
        sequence_length = tf.shape(token_ids)[1]
        positions = tf.range(sequence_length)

        token_embeddings = self.embedding_layer(token_ids)
        position_embeddings = self.positional_encoding(positions)

        x = token_embeddings + position_embeddings
        x = apply_transformer_stack(x, self.transformer_blocks)
        
        logits = self.output_layer(x)
        return logits