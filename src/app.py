#uvicorn app:app --host 0.0.0.0 --port 8000
from env_loader import load_backend_env

load_backend_env()

from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, status, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import httpx

from database import get_db, Connection
from services.user_service import UserService
from services.email_verification_service import (
    generate_email_verification_code,
    send_email_verification_email,
    send_password_reset_email,
)
from models.user import User, UserCreate
from models.team import TeamCreate
from rbac import derive_app_role, permissions_for_role
from services.team_service import TeamService

import os
from google.oauth2 import id_token
from google.auth.transport import requests

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
from controllers.tracking_profile_controller import router as tracking_profile_router

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
app.include_router(tracking_profile_router, prefix="/api", tags=["Tracking Profiles"])

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

def generate_user_token_data(user, db: Connection):
    from rbac import derive_app_role, permissions_for_role
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
            app_role = derive_app_role(user.user_type, None)
            token_data.update({
                "app_role": app_role,
                "app_permissions": permissions_for_role(app_role),
            })
        else:
            app_role = derive_app_role(user.user_type, staff.role)
            token_data.update({
                "staff_id": staff.id,
                "team_id": staff.team_id,
                "permission_level": staff.permission_level,
                "role": staff.role,
                "app_role": app_role,
                "app_permissions": permissions_for_role(app_role),
            })
    return token_data


def _derive_default_team_name(full_name: Optional[str], email: str) -> str:
    preferred_name = (full_name or "").strip()
    if not preferred_name:
        local_part = email.split("@", 1)[0]
        preferred_name = " ".join(
            segment.capitalize()
            for segment in local_part.replace(".", " ").replace("_", " ").replace("-", " ").split()
        ) or "My"

    team_name = f"{preferred_name}'s team"
    return team_name[:50]


def _provision_default_team_for_user(
    db: Connection,
    user: User,
    *,
    full_name: Optional[str] = None,
) -> None:
    team_service = TeamService(db)
    existing_teams = team_service.get_all_teams(user.id)
    if existing_teams:
        return

    default_team = TeamCreate(name=_derive_default_team_name(full_name, user.email))
    team_service.create_team(default_team, user.id)


def _create_user_with_default_team(
    db: Connection,
    user_create: UserCreate,
) -> User:
    user_service = UserService(db)
    new_user = user_service.create_user(user_create)

    try:
        _provision_default_team_for_user(
            db,
            new_user,
            full_name=user_create.full_name,
        )
    except Exception:
        user_service.delete_user(new_user.id)
        raise

    return new_user


def _issue_access_token_for_user(user: User, db: Connection) -> Dict[str, str]:
    token_data = generate_user_token_data(user, db)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


def _finalize_verified_user(
    db: Connection,
    user: User,
    *,
    full_name: Optional[str] = None,
) -> User:
    if user.user_type == "owner":
        _provision_default_team_for_user(db, user, full_name=full_name or user.full_name)

    user_service = UserService(db)
    return user_service.activate_verified_user(user.id) or user


def _verification_expiry_utc() -> datetime:
    ttl_minutes = int(os.environ.get("EMAIL_VERIFICATION_CODE_TTL_MINUTES", "15"))
    return datetime.utcnow() + timedelta(minutes=ttl_minutes)

@app.post("/api/token", tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Connection = Depends(get_db)):
    user_service = UserService(db)
    user = user_service.get_user_by_email(email=form_data.username)

    # Case 1: User not found
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
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

    if not user.is_active:
        if user_service.is_email_verification_pending(user.email):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Email not verified. Enter the verification code sent to your inbox.",
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is inactive.",
        )

    # Case 3: Success
    return _issue_access_token_for_user(user, db)

from pydantic import BaseModel
class GoogleAuthRequest(BaseModel):
    id_token: Optional[str] = None
    access_token: Optional[str] = None


class RegisterVerifyRequest(BaseModel):
    email: str
    code: str


class RegisterResendRequest(BaseModel):
    email: str


class PasswordResetRequest(BaseModel):
    email: str


class PasswordResetConfirmRequest(BaseModel):
    email: str
    code: str
    new_password: str


async def _fetch_google_userinfo_from_access_token(access_token: str) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=502,
            detail="Google userinfo request timed out",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Google userinfo request failed",
        ) from exc

    if response.status_code in {400, 401, 403}:
        raise HTTPException(status_code=401, detail="Invalid Google access token")
    if response.status_code >= 500:
        raise HTTPException(
            status_code=502,
            detail="Google userinfo service unavailable",
        )
    if response.status_code >= 300:
        raise HTTPException(status_code=401, detail="Invalid Google access token")

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Invalid Google userinfo response",
        ) from exc

    if not payload.get("sub") or not payload.get("email"):
        raise HTTPException(status_code=401, detail="Invalid Google access token")

    email_verified = payload.get("email_verified")
    if email_verified not in (True, "true", "True", 1, "1"):
        raise HTTPException(
            status_code=401,
            detail="Google account email is not verified",
        )

    return payload

@app.post("/api/auth/google", tags=["Authentication"])
async def google_auth(auth_req: GoogleAuthRequest, db: Connection = Depends(get_db)):
    user_service = UserService(db)
    if auth_req.access_token:
        id_info = await _fetch_google_userinfo_from_access_token(auth_req.access_token)
        email = id_info["email"]
        full_name = id_info.get("name")
    elif auth_req.id_token:
        allowed_google_client_ids = [
            value.strip()
            for value in {
                os.environ.get("GOOGLE_SERVER_CLIENT_ID", ""),
                os.environ.get("GOOGLE_WEB_CLIENT_ID", ""),
                os.environ.get("GOOGLE_CLIENT_ID", ""),
            }
            if value and value.strip()
        ]

        if not allowed_google_client_ids:
            raise HTTPException(
                status_code=500,
                detail="Google client ID not configured on server",
            )

        try:
            id_info = id_token.verify_oauth2_token(
                auth_req.id_token,
                requests.Request(),
                audience=None,
            )
            audience = id_info.get("aud")
            if audience not in allowed_google_client_ids:
                raise ValueError("Invalid Google token audience")

            email = id_info["email"]
            full_name = id_info.get("name")
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid Google token")
    else:
        raise HTTPException(
            status_code=400,
            detail="Google authentication requires access_token or id_token",
        )

    # Check if user exists
    user = user_service.get_user_by_email(email=email)
    
    if not user:
        from models.user import UserCreate
        user_create = UserCreate(
            email=email,
            full_name=full_name,
            user_type="owner",
            is_active=True
        )
        try:
            user = _create_user_with_default_team(db, user_create)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not provision the default team for this account.",
            ) from exc
    elif not user_service.is_email_verified(user.id):
        try:
            user = _finalize_verified_user(db, user, full_name=full_name)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not finalize this account after Google verification.",
            ) from exc

    # Generate our JWT
    return _issue_access_token_for_user(user, db)

@app.post("/api/register", tags=["Authentication"])
async def register_user(user_create: UserCreate, db: Connection = Depends(get_db)):
    user_service = UserService(db)
    existing_user = user_service.get_user_by_email(email=user_create.email)
    if existing_user:
        if user_service.is_email_verification_pending(existing_user.email):
            verification_code = generate_email_verification_code()
            verification_expires_at = _verification_expiry_utc()
            user_service.refresh_email_verification_code(
                existing_user.email,
                verification_code=verification_code,
                verification_expires_at=verification_expires_at,
            )
            try:
                send_email_verification_email(
                    existing_user.email,
                    verification_code,
                    full_name=existing_user.full_name,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Could not send the verification email.",
                ) from exc
            return {
                "status": "pending_verification",
                "email": existing_user.email,
                "detail": "Verification code re-sent.",
            }
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    verification_code = generate_email_verification_code()
    verification_expires_at = _verification_expiry_utc()

    try:
        pending_user = user_service.create_pending_user(
            user_create,
            verification_code=verification_code,
            verification_expires_at=verification_expires_at,
        )
        send_email_verification_email(
            pending_user.email,
            verification_code,
            full_name=pending_user.full_name,
        )
    except Exception as exc:
        if "pending_user" in locals():
            user_service.delete_user(pending_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not start email verification for this account.",
        ) from exc

    return {
        "status": "pending_verification",
        "email": pending_user.email,
        "detail": "Verification code sent.",
    }


@app.post("/api/register/verify", tags=["Authentication"])
async def verify_registered_user(payload: RegisterVerifyRequest, db: Connection = Depends(get_db)):
    user_service = UserService(db)
    user = user_service.verify_email_code(payload.email, payload.code)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code.",
        )

    if not user_service.is_email_verified(user.id):
        try:
            user = _finalize_verified_user(db, user, full_name=user.full_name)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not finish account verification.",
            ) from exc

    return _issue_access_token_for_user(user, db)


@app.post("/api/register/resend", tags=["Authentication"])
async def resend_registration_code(payload: RegisterResendRequest, db: Connection = Depends(get_db)):
    user_service = UserService(db)
    user = user_service.get_user_by_email(payload.email)
    if not user or user_service.is_email_verified(user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pending email verification found for this account.",
        )

    verification_code = generate_email_verification_code()
    verification_expires_at = _verification_expiry_utc()
    updated = user_service.refresh_email_verification_code(
        user.email,
        verification_code=verification_code,
        verification_expires_at=verification_expires_at,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No pending email verification found for this account.",
        )

    try:
        send_email_verification_email(
            user.email,
            verification_code,
            full_name=user.full_name,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not send the verification email.",
        ) from exc

    return {
        "status": "pending_verification",
        "email": user.email,
        "detail": "Verification code re-sent.",
    }


@app.post("/api/password/forgot", tags=["Authentication"])
async def request_password_reset(payload: PasswordResetRequest, db: Connection = Depends(get_db)):
    user_service = UserService(db)
    reset_code = generate_email_verification_code()
    reset_expires_at = _verification_expiry_utc()

    user = user_service.start_password_reset(
        payload.email,
        reset_code=reset_code,
        reset_expires_at=reset_expires_at,
    )

    if user:
        try:
            send_password_reset_email(
                user.email,
                reset_code,
                full_name=user.full_name,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not send the password reset email.",
            ) from exc

    return {
        "status": "ok",
        "detail": "If the account exists, a password reset code was sent.",
    }


@app.post("/api/password/reset", tags=["Authentication"])
async def reset_password_with_code(payload: PasswordResetConfirmRequest, db: Connection = Depends(get_db)):
    if len(payload.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long.",
        )

    user_service = UserService(db)
    user = user_service.reset_password_with_code(
        payload.email,
        reset_code=payload.code,
        new_password=payload.new_password,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired password reset code.",
        )

    return {
        "status": "ok",
        "detail": "Password updated successfully.",
    }

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Football Match Analysis API"}
