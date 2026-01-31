#uvicorn app:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, status, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from database import get_db, Connection
from database import get_db, Connection
from services.user_service import UserService
from models.user import User, UserCreate

import cv2
import numpy as np
import tempfile
import os
import uuid

# --- Configuration for JWT --- #
# TODO: Use environment variables for these in production
SECRET_KEY = "your-secret-key" # openssl rand -hex 32
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Connection = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
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

# Import controllers
from controllers.team_controller import router as team_router
from controllers.user_controller import router as user_router
from controllers.player_controller import router as player_router
from controllers.staff_controller import router as staff_router
from controllers.match_controller import router as match_router
from controllers.analysis_report_controller import router as analysis_report_router
from controllers.match_event_controller import router as match_event_router
from controllers.player_match_statistics_controller import router as player_match_statistics_router
from controllers.formation_controller import router as formation_router
from controllers.match_lineup_controller import router as match_lineup_router
from controllers.video_segment_controller import router as video_segment_router
from controllers.reunion_controller import router as reunion_router
from controllers.training_session_controller import router as training_session_router
from controllers.event_controller import router as event_router # New import
from controllers.match_team_statistics_controller import router as match_team_statistics_router # New import

app = FastAPI(
    title="Football Match Analysis API",
    description="API for analyzing football match videos using YOLOv8 and managing related data.",
    version="1.0.0",
)

# Mount the static directory to serve images
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- CORS Middleware ---
# Allow requests from frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(team_router, prefix="/api", tags=["Teams"])
app.include_router(user_router, prefix="/api", tags=["Users"])
app.include_router(player_router, prefix="/api", tags=["Players"])
app.include_router(staff_router, prefix="/api", tags=["Staff"])
app.include_router(match_router, prefix="/api", tags=["Matches"])
app.include_router(analysis_report_router, prefix="/api", tags=["Analysis Reports"])
app.include_router(match_event_router, prefix="/api", tags=["Match Events"])
app.include_router(player_match_statistics_router, prefix="/api", tags=["Player Match Statistics"])
app.include_router(formation_router, prefix="/api", tags=["Formations"])
app.include_router(match_lineup_router, prefix="/api", tags=["Match Lineups"])
app.include_router(video_segment_router, prefix="/api", tags=["Video Segments"])
app.include_router(reunion_router, prefix="/api", tags=["Reunions"])
app.include_router(training_session_router, prefix="/api", tags=["Training Sessions"])
app.include_router(event_router, prefix="/api", tags=["Events"]) # New router
app.include_router(match_team_statistics_router, prefix="/api", tags=["Match Team Statistics"]) # New router

# Initialize the FootballAnalyzer for single image analysis (can be reused)
# MOVED to analysis_app.py

# In-memory store for analysis status
# MOVED to analysis_app.py

# Analysis status endpoint moved to analysis_app.py

@app.post("/api/token", tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Connection = Depends(get_db)):
    user_service = UserService(db)
    user = user_service.get_user_by_email(email=form_data.username)

    # Case 1: User not found
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, # Or 404, but 401 is better for security
            detail="This account does not exist.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Case 2: Incorrect password
    if not user_service.verify_password(form_data.password, str(user.password_hash)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Case 3: Success
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/register", tags=["Authentication"])
async def register_user(user_create: UserCreate, db: Connection = Depends(get_db)):
    user_service = UserService(db)
    existing_user = user_service.get_user_by_email(email=user_create.email)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    new_user = user_service.create_user(user_create)

    # Generate access token for immediate login
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Football Match Analysis API"}

# Image detection endpoint moved to analysis_app.py

# Match analysis endpoint moved to analysis_app.py

# --- Authentication Example ---
# If your frontend uses authentication, add dependencies to endpoints:
# from fastapi.security import OAuth2PasswordBearer
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# @app.get("/secure-endpoint")
# async def secure_endpoint(token: str = Depends(oauth2_scheme)):
#     # ...authentication logic...
#     pass
