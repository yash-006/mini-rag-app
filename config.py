
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Cohere configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validate required credentials
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL not found in .env file")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY not found in .env file")
