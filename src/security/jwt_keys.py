import os
from pathlib import Path


def _project_root() -> Path:
    # backend/src/security/jwt_keys.py -> backend/src -> backend
    return Path(__file__).resolve().parents[2]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_pem_from_env_or_file(*, env_pem: str, env_path: str, default_path: Path) -> str:
    """
    Load a PEM string from:
    1) env var containing the PEM itself (preferred for prod)
    2) env var containing a file path
    3) default path on disk (dev convenience)
    """
    pem = os.environ.get(env_pem, "").strip()
    if pem:
        return pem

    path_raw = os.environ.get(env_path, "").strip()
    if path_raw:
        return _read_text_file(Path(path_raw).expanduser().resolve())

    return _read_text_file(default_path)


def get_jwt_private_key() -> str:
    root = _project_root()
    return load_pem_from_env_or_file(
        env_pem="JWT_PRIVATE_KEY",
        env_path="JWT_PRIVATE_KEY_PATH",
        default_path=root / "certs" / "private.pem",
    )


def get_jwt_public_key() -> str:
    root = _project_root()
    return load_pem_from_env_or_file(
        env_pem="JWT_PUBLIC_KEY",
        env_path="JWT_PUBLIC_KEY_PATH",
        default_path=root / "certs" / "public.pem",
    )

