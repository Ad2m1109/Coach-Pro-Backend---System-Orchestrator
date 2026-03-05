from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os

from database import get_db, Connection
from services.user_service import UserService
from models.user import User
from rbac import APP_ROLE_ACCOUNT_MANAGER

# --- Configuration for JWT (RS256) --- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIVATE_KEY_PATH = os.path.join(PROJECT_ROOT, "certs", "private.pem")
PUBLIC_KEY_PATH = os.path.join(PROJECT_ROOT, "certs", "public.pem")
if not os.path.exists(PRIVATE_KEY_PATH):
    # Fallback for local dev if PROJECT_ROOT logic fails in some shell envs
    PRIVATE_KEY_PATH = "certs/private.pem"
    PUBLIC_KEY_PATH = "certs/public.pem"

with open(PRIVATE_KEY_PATH, "r") as f:
    RSA_PRIVATE_KEY = f.read()

with open(PUBLIC_KEY_PATH, "r") as f:
    RSA_PUBLIC_KEY = f.read()

ALGORITHM = "RS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, RSA_PRIVATE_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Connection = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, RSA_PUBLIC_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user_service = UserService(db)
    user = user_service.get_user_by_email(email=email)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    request: Request,
    current_user: User = Depends(get_current_user),
    token: str = Depends(oauth2_scheme),
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    # The AI assistant is accessible to all active authenticated users
    if request.url.path.startswith("/api/assistant"):
        return current_user

    app_role = _get_app_role_from_token(token)
    path = request.url.path
    method = request.method.upper()
    write_methods = {"POST", "PUT", "PATCH", "DELETE"}
    manager_only_write_prefixes = (
        "/api/users",
        "/api/staff",
        "/api/teams",
    )

    if app_role == APP_ROLE_ACCOUNT_MANAGER:
        if path.startswith("/api/"):
            # Account manager can view all modules.
            if method in ("GET", "HEAD", "OPTIONS"):
                return current_user

            # Account manager can modify team + account resources.
            if method in write_methods and path.startswith(manager_only_write_prefixes):
                return current_user

            raise HTTPException(
                status_code=403,
                detail="Account manager can view all data but can only modify team and account resources",
            )
    else:
        if method in write_methods and path.startswith(manager_only_write_prefixes):
            raise HTTPException(
                status_code=403,
                detail="Only account manager can modify team and account resources",
            )
    return current_user


def _decode_token_payload(token: str) -> dict:
    try:
        return jwt.decode(token, RSA_PUBLIC_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _get_app_role_from_token(token: str) -> str:
    payload = _decode_token_payload(token)
    return payload.get("app_role") or ("account_manager" if payload.get("user_type") == "owner" else "player")


async def require_account_manager(
    current_user: User = Depends(get_current_active_user),
    token: str = Depends(oauth2_scheme),
):
    if _get_app_role_from_token(token) != APP_ROLE_ACCOUNT_MANAGER:
        raise HTTPException(status_code=403, detail="Only account manager can perform this action")
    return current_user
