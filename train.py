from src.config_loader import load_config
from src.logger import setup_logging
from src.data_handling.data_loader import load_txt_data, split_train_val
from src.cleaning.text_cleaner import clean_text
from src.tokenizer.tokenier_bpe import get_tokens, encode
from src.model.gpt_model import GPTModel
from src.model.dataset import create_batched_dataset
from src.training import train_model

config = load_config("config/config.yaml")

setup_logging(
    log_level=config["logging"]["log_level"],
    log_file=config["logging"]["log_file"]
)

raw_text = load_txt_data(config["data"]["file_path"])
cleaned_text = clean_text(raw_text)

train_text, val_text = split_train_val(
    cleaned_text,
    val_ratio=config["data"]["val_ratio"]
)

tokenizer = get_tokens(config["tokenizer"]["encoding_name"])
train_token_ids = encode(train_text, tokenizer)

train_dataset = create_batched_dataset(
    train_token_ids,
    batch_size=config["train"]["batch_size"],
    max_sequence_length=config["model"]["max_sequence_length"]
)

model = GPTModel(
    vocab_size=config["model"]["vocab_size"],
    embedding_dim=config["model"]["embedding_dim"],
    max_sequence_length=config["model"]["max_sequence_length"],
    num_heads=config["model"]["num_heads"],
    hidden_dim=config["model"]["hidden_dim"],
    num_blocks=config["model"]["num_blocks"],
    activation=config["model"]["activation"]
)

train_model(
    model=model,
    dataset=train_dataset,
    epochs=config["train"]["num_epochs"],
    learning_rate=config["train"]["learning_rate"]
)