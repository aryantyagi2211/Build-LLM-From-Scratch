from datasets import load_dataset
import logging

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
logger = logging.getLogger(__name__)


def download_wikitext(output_path, dataset_name="Salesforce/wikitext", dataset_config="wikitext-2-raw-v1"):
    try:
        dataset = load_dataset(dataset_name, dataset_config)
    except Exception as error:
        logger.error(f"failed to download {dataset_name} dataset: {error}")
        raise
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for row in dataset["train"]:
                if row["text"].strip():
                    f.write(row["text"])
    except (IOError, OSError) as error:
        logger.error(f"failed to write {output_path} dataset: {error}")

    logger.info(f"Download {dataset_config} and saved to {output_path}")