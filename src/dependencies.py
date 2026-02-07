from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os

from database import get_db, Connection
from services.user_service import UserService
from models.user import User

# --- Configuration for JWT (RS256) --- #
if os.path.exists("certs/private.pem"):
    PRIVATE_KEY_PATH = os.path.join("certs", "private.pem")
    PUBLIC_KEY_PATH = os.path.join("certs", "public.pem")
elif os.path.exists("../certs/private.pem"):
    PRIVATE_KEY_PATH = os.path.join("../certs", "private.pem")
    PUBLIC_KEY_PATH = os.path.join("../certs", "public.pem")
else:
    PRIVATE_KEY_PATH = "/home/ademyoussfi/Desktop/Projects/football-coach/backend/certs/private.pem"
    PUBLIC_KEY_PATH = "/home/ademyoussfi/Desktop/Projects/football-coach/backend/certs/public.pem"

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

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
