"""Encryption-at-rest for uploaded medical files and upload sniffing."""
import base64, hashlib
from cryptography.fernet import Fernet, InvalidToken
from flask import current_app


def _fernet():
    key = current_app.config.get("FILE_ENCRYPTION_KEY")
    if not key:  # derive a stable key from SECRET_KEY (set FILE_ENCRYPTION_KEY in production)
        key = base64.urlsafe_b64encode(hashlib.sha256(current_app.config["SECRET_KEY"].encode()).digest())
    return Fernet(key)


def encrypt_bytes(data: bytes) -> bytes:
    return _fernet().encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    try:
        return _fernet().decrypt(token)
    except InvalidToken as exc:
        raise ValueError("Stored file could not be decrypted (wrong key?).") from exc


def sniff_type(data: bytes):
    """Identify an upload from its magic bytes (never trust the extension). Returns pdf/png/jpg or None."""
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    return None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
