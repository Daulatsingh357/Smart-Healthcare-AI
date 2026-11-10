"""Application configuration. Secrets come from environment variables / .env."""
import os
from dotenv import load_dotenv

load_dotenv()
BASE = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") or "sqlite:///" + os.path.join(BASE, "database", "healthcare.db")
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}
    UPLOAD_FOLDER = os.path.join(BASE, "uploads")
    MODEL_DIR = os.getenv("MODEL_DIR") or os.path.join(BASE, "ml_models", "saved_models")
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024          # 8 MB upload limit
    ALLOWED_EXT = {"pdf", "jpg", "jpeg", "png"}
    # Optional: set FILE_ENCRYPTION_KEY (a Fernet key) to encrypt uploads with a dedicated key.
    FILE_ENCRYPTION_KEY = os.getenv("FILE_ENCRYPTION_KEY", "")
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")  # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    WTF_CSRF_TIME_LIMIT = None


class TestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SECRET_KEY = "test-secret"
