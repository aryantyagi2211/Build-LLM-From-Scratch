from src.config_loader import load_config
from src.logger import setup_logging
from src.tokenizer.tokenier_bpe import get_tokens
from src.model.gpt_model import GPTModel
from src.inference import generate_text

config = load_config("config/config.yaml")


setup_logging(
    log_level=config["logging"]["log_level"],
    log_file=config["logging"]["log_file"]
)

tokenizer = get_tokens(config["tokenizer"]["encoding_name"])

model = GPTModel(
    vocab_size=config["model"]["vocab_size"],
    embedding_dim=config["model"]["embedding_dim"],
    max_sequence_length=config["model"]["max_sequence_length"],
    num_heads=config["model"]["num_heads"],
    hidden_dim=config["model"]["hidden_dim"],
    num_blocks=config["model"]["num_blocks"],
    activation=config["model"]["activation"]
)

dummy_input = tf.constant([[0] * config["model"]["max_sequence_length"]])
model(dummy_input)

model.load_weights("checkpoints/model_epoch_02_weights.h5")

generated = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=config["inference"]["prompt"],
    max_new_tokens=config["inference"]["max_new_tokens"],
    max_sequence_length=config["model"]["max_sequence_length"]
)

print("Generated Text:", generated)