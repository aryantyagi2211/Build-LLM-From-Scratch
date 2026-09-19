import logging

from src.model.attention import build_attention_model
from src.model.feed_forward import feed_forward
from src.model.layer_norm import build_layer_norm

logger = logging.getLogger(__name__)


def build_transformer_block(embedding_dim, num_heads, hidden_dim, activation):
    block = {
        "attention_layer": build_attention_model(embedding_dim=embedding_dim, num_heads=num_heads),
        "feed_forward_layer": feed_forward(embedding_dim=embedding_dim, hidden_dim=hidden_dim, activation=activation),
        "layer_norm_1": build_layer_norm(),
        "layer_norm_2": build_layer_norm(),
    }
    return block


def apply_transformer_block(x, block):
    normalized_x = block["layer_norm_1"](x)
    attention_output = block["attention_layer"](
        query=normalized_x, key=normalized_x, value=normalized_x, use_causal_mask=True
    )
    x = x + attention_output

    normalized_x = block["layer_norm_2"](x)
    feed_forward_output = block["feed_forward_layer"](normalized_x)
    x = x + feed_forward_output

    return x


def build_transformer_stack(embedding_dim, num_heads, hidden_dim, activation, num_blocks):
    blocks = [
        build_transformer_block(embedding_dim, num_heads, hidden_dim, activation)
        for _ in range(num_blocks)
    ]
    logger.info(f"Transformer stack created with {num_blocks} blocks")
    return blocks


def apply_transformer_stack(x, blocks):
    for block in blocks:
        x = apply_transformer_block(x, block)
    return x