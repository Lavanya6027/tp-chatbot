from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")