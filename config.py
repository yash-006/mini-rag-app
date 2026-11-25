import os

from dotenv import load_dotenv

# Load variables from the local .env file (ignored if it doesn't exist)
load_dotenv()


def get_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)

    if not value:
        raise ValueError(
            f"{name} not found. Add it to your .env file "
            f"or to your hosting platform's secrets."
        )

    return value