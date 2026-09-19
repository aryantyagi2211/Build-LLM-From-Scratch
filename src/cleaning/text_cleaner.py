import re
import logging

logger = logging.getLogger(__name__)

def clean_text(raw_text):
    text=re.sub(r'^\s*=+.*=+\s*$', '', raw_text, flags=re.MULTILINE)
    text=re.sub(r' +', '', text)
    text=re.sub(r'\ns*\n+', '\n', text)
    text=text.strip()

    logger.info(f"cleaning text: {len(raw_text)} chars -> {len(text)} chars")
    return text
