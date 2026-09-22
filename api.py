import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config_loader import load_config
from src.logger import setup_logging
from src.tokenizer.tokenier_bpe import get_tokens
from src.model.gpt_model import GPTModel
from src.model.checkpoint_utils import load_checkpoint_if_exists
from src.inference import generate_text
from src.guardrails import validate_input, validate_output

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

model, _ = load_checkpoint_if_exists(
    model=model,
    checkpoint_dir=config["train"]["checkpoint_dir"],
    max_sequence_length=config["model"]["max_sequence_length"]
)

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 20

@app.post("/generate")
def generate(request: GenerateRequest):
    try: 
        validate_input(request.prompt)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        max_sequence_length=config["model"]["max_sequence_length"]
    )
    try:
        validate_output(generated_text)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return {"generated_text": generated_text}