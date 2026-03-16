#uvicorn app:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, status, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from database import get_db, Connection
from services.user_service import UserService
from models.user import User, UserCreate
from rbac import derive_app_role, permissions_for_role

import os

from dependencies import (
    create_access_token, 
    get_current_user, 
    get_current_active_user,
    RSA_PRIVATE_KEY,
    RSA_PUBLIC_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    oauth2_scheme
)

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
from controllers.note_controller import router as note_router
from controllers.analysis_controller import router as analysis_router
from controllers.assistant_controller import router as assistant_router
from controllers.tactical_alert_controller import router as tactical_alert_router
from controllers.segment_controller import router as segment_router
# from controllers.simulation_controller import router as simulation_router

app = FastAPI(
    title="Football Match Analysis API",
    description="API for analyzing football match videos using YOLOv8 and managing related data.",
    version="1.0.0",
)

# Mount the static directory to serve images
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- CORS Middleware ---
# Allow requests from frontend (adjust origins as needed)
cors_origins_raw = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
cors_origins = (
    [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
    if cors_origins_raw
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    # If origins is '*', credentials must be disabled (browsers reject '*' + credentials).
    allow_credentials=False if cors_origins == ["*"] else True,
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
app.include_router(event_router, prefix="/api", tags=["Events"])
app.include_router(match_team_statistics_router, prefix="/api", tags=["Match Team Statistics"])
app.include_router(note_router, prefix="/api", tags=["Notes"])
app.include_router(analysis_router, prefix="/api", tags=["Analysis"])
app.include_router(assistant_router, prefix="/api/assistant", tags=["AI Assistant"])
app.include_router(tactical_alert_router, tags=["Tactical Alerts"])
app.include_router(segment_router, prefix="/api", tags=["Analysis Segments"])

from fastapi import WebSocket, WebSocketDisconnect
from controllers.tactical_alert_controller import alert_service

@app.websocket("/ws/alerts/{match_id}")
async def tactical_alerts_websocket(websocket: WebSocket, match_id: str):
    await alert_service.connect(match_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        alert_service.disconnect(match_id, websocket)
    except Exception:
        alert_service.disconnect(match_id, websocket)

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

    # Case 3: Success - Build token claims based on user type
    token_data = {
        "sub": user.email,
        "user_type": user.user_type,
    }

    if user.user_type == "owner":
        app_role = derive_app_role(user.user_type, None)
        token_data.update({
            "app_role": app_role,
            "app_permissions": permissions_for_role(app_role),
        })
    
    elif user.user_type == "staff":
        # Get staff record for permission level
        from services.staff_service import StaffService
        staff_service = StaffService(db)
        staff = staff_service.get_staff_by_user_id(user.id)
        
        if not staff:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Staff record not found",
            )
        
        app_role = derive_app_role(user.user_type, staff.role)
        token_data.update({
            "staff_id": staff.id,
            "team_id": staff.team_id,
            "permission_level": staff.permission_level,
            "role": staff.role,
            "app_role": app_role,
            "app_permissions": permissions_for_role(app_role),
        })

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data=token_data, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/register", tags=["Authentication"])
async def register_user(user_create: UserCreate, db: Connection = Depends(get_db)):
    user_service = UserService(db)
    if user_service.count_users() > 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Self-registration is disabled. Ask account manager to create your account.",
        )
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
