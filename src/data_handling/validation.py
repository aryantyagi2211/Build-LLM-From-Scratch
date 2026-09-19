import os

def validate_data_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist")

    if os.path.getsize(filepath) == 0:
        raise ValueError(f"File {filepath} is empty")