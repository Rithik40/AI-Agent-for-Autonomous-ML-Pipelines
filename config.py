"""
Configuration management – loads environment variables from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Feature selection threshold: only apply if feature count exceeds this
FEATURE_SELECTION_THRESHOLD: int = 15

# Cross-validation folds
CV_FOLDS: int = 5

# Temp directory for downloaded CSVs and generated plots/PDFs
TEMP_DIR: str = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

def validate_config() -> None:
    """Raise if required env vars are missing."""
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Please create a .env file. See .env.example for reference."
        )
