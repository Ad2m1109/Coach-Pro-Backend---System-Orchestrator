import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import pymysql
import pytest
from fastapi import HTTPException


BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


class _ImportTimeCursor:
    def execute(self, sql, params=None):
        self._last_sql = sql
        self._last_params = params

    def fetchone(self):
        return {"count": 1}

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ImportTimeConnection:
    def cursor(self):
        return _ImportTimeCursor()

    def commit(self):
        pass

    def close(self):
        pass


_original_pymysql_connect = pymysql.connect
pymysql.connect = lambda **kwargs: _ImportTimeConnection()

_original_cwd = os.getcwd()
os.chdir(SRC_DIR)
try:
    import app as backend_app
finally:
    os.chdir(_original_cwd)
    pymysql.connect = _original_pymysql_connect

from models.user import User, UserCreate, UserRead, UserTypeEnum


def _owner_user() -> User:
    return User(
        id="user-1",
        email="owner@example.com",
        full_name="Owner",
        user_type=UserTypeEnum.owner,
        is_active=True,
        created_at=datetime.utcnow(),
        last_login=None,
        password_hash="hashed",
    )


def _staff_user() -> User:
    return User(
        id="staff-user-1",
        email="coach@example.com",
        full_name="Coach User",
        user_type=UserTypeEnum.staff,
        is_active=True,
        created_at=datetime.utcnow(),
        last_login=None,
        password_hash="hashed",
    )


def _pending_owner_user() -> User:
    return User(
        id="pending-user-1",
        email="owner@example.com",
        full_name="Owner",
        user_type=UserTypeEnum.owner,
        is_active=False,
        created_at=datetime.utcnow(),
        last_login=None,
        password_hash="hashed",
    )


def test_register_user_starts_email_verification(monkeypatch):
    pending_user = _pending_owner_user()
    captured = {}

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def count_users(self):
            return 0

        def get_user_by_email(self, email):
            return None

        def create_pending_user(self, user_create, *, verification_code, verification_expires_at):
            captured["created_user_input"] = user_create
            captured["verification_code"] = verification_code
            captured["verification_expires_at"] = verification_expires_at
            return pending_user

    monkeypatch.setattr(backend_app, "UserService", FakeUserService)
    monkeypatch.setattr(
        backend_app,
        "generate_email_verification_code",
        lambda: "654321",
    )
    monkeypatch.setattr(
        backend_app,
        "send_email_verification_email",
        lambda email, code, full_name=None: captured.update(
            {
                "sent_email": email,
                "sent_code": code,
                "sent_full_name": full_name,
            }
        ),
    )

    result = asyncio.run(
        backend_app.register_user(
            UserCreate(
                email="owner@example.com",
                password="secret123",
                full_name="Owner",
            ),
            db=object(),
        )
    )

    assert result["status"] == "pending_verification"
    assert result["email"] == "owner@example.com"
    assert captured["created_user_input"].email == "owner@example.com"
    assert captured["verification_code"] == "654321"
    assert captured["sent_email"] == "owner@example.com"
    assert captured["sent_code"] == "654321"
    assert captured["sent_full_name"] == "Owner"


def test_verify_registered_user_issues_token_and_provisions_team(monkeypatch):
    pending_user = _pending_owner_user()
    verified_user = _owner_user()
    captured = {"is_email_verified_calls": 0}

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def verify_email_code(self, email, code):
            captured["verify_email"] = email
            captured["verify_code"] = code
            return pending_user

        def is_email_verified(self, user_id):
            captured["is_email_verified_calls"] += 1
            return captured["is_email_verified_calls"] > 1

        def activate_verified_user(self, user_id):
            captured["activated_user_id"] = user_id
            return verified_user

    class FakeTeamService:
        def __init__(self, db):
            self.db = db

        def get_all_teams(self, user_id):
            captured["team_lookup_user_id"] = user_id
            return []

        def create_team(self, team_create, user_id):
            captured["created_team_name"] = team_create.name
            captured["created_team_user_id"] = user_id
            return {"id": "team-1"}

    monkeypatch.setattr(backend_app, "UserService", FakeUserService)
    monkeypatch.setattr(backend_app, "TeamService", FakeTeamService)
    monkeypatch.setattr(
        backend_app,
        "create_access_token",
        lambda *, data, expires_delta: "signed-token",
    )

    result = asyncio.run(
        backend_app.verify_registered_user(
            backend_app.RegisterVerifyRequest(
                email="owner@example.com",
                code="654321",
            ),
            db=object(),
        )
    )

    assert result == {"access_token": "signed-token", "token_type": "bearer"}
    assert captured["verify_email"] == "owner@example.com"
    assert captured["verify_code"] == "654321"
    assert captured["activated_user_id"] == pending_user.id
    assert captured["created_team_user_id"] == pending_user.id
    assert captured["created_team_name"] == "Owner's team"


def test_password_reset_request_sends_code_for_existing_verified_user(monkeypatch):
    created_user = _owner_user()
    captured = {}

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def start_password_reset(self, email, *, reset_code, reset_expires_at):
            captured["email"] = email
            captured["reset_code"] = reset_code
            captured["reset_expires_at"] = reset_expires_at
            return created_user

    monkeypatch.setattr(backend_app, "UserService", FakeUserService)
    monkeypatch.setattr(
        backend_app,
        "generate_email_verification_code",
        lambda: "998877",
    )
    monkeypatch.setattr(
        backend_app,
        "send_password_reset_email",
        lambda email, code, full_name=None: captured.update(
            {
                "sent_email": email,
                "sent_code": code,
                "sent_full_name": full_name,
            }
        ),
    )

    result = asyncio.run(
        backend_app.request_password_reset(
            backend_app.PasswordResetRequest(email="owner@example.com"),
            db=object(),
        )
    )

    assert result["status"] == "ok"
    assert captured["email"] == "owner@example.com"
    assert captured["reset_code"] == "998877"
    assert captured["sent_email"] == "owner@example.com"
    assert captured["sent_code"] == "998877"
    assert captured["sent_full_name"] == "Owner"


def test_password_reset_with_code_updates_password(monkeypatch):
    updated_user = _owner_user()
    captured = {}

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def reset_password_with_code(self, email, *, reset_code, new_password):
            captured["email"] = email
            captured["reset_code"] = reset_code
            captured["new_password"] = new_password
            return updated_user

    monkeypatch.setattr(backend_app, "UserService", FakeUserService)

    result = asyncio.run(
        backend_app.reset_password_with_code(
            backend_app.PasswordResetConfirmRequest(
                email="owner@example.com",
                code="998877",
                new_password="new-secret123",
            ),
            db=object(),
        )
    )

    assert result["status"] == "ok"
    assert captured["email"] == "owner@example.com"
    assert captured["reset_code"] == "998877"
    assert captured["new_password"] == "new-secret123"


def test_google_auth_auto_provisions_user_when_email_not_found(monkeypatch):
    captured = {}

    class FakeUserService:
        def __init__(self, db):
            self.db = db
            self.created_user = _owner_user()
            self.lookup_email = None
            self.created_input = None

        def get_user_by_email(self, email):
            self.lookup_email = email
            return None

        def count_users(self):
            return 1

        def create_user(self, user_create):
            self.created_input = user_create
            return self.created_user

        def delete_user(self, user_id):
            captured["deleted_user_id"] = user_id
            return True

    class FakeTeamService:
        def __init__(self, db):
            self.db = db

        def get_all_teams(self, user_id):
            captured["team_lookup_user_id"] = user_id
            return []

        def create_team(self, team_create, user_id):
            captured["created_team_input"] = team_create
            captured["created_team_user_id"] = user_id
            return {"id": "team-1"}

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "legacy-client-id")
    monkeypatch.setenv("GOOGLE_WEB_CLIENT_ID", "web-client-id")
    monkeypatch.setenv("GOOGLE_SERVER_CLIENT_ID", "server-client-id")
    fake_service = FakeUserService(object())
    monkeypatch.setattr(backend_app, "UserService", lambda db: fake_service)
    monkeypatch.setattr(backend_app, "TeamService", FakeTeamService)
    async def fake_fetch_userinfo(access_token):
        assert access_token == "mobile-access-token"
        return {
            "sub": "google-sub",
            "email": "new-google-user@example.com",
            "name": "Google User",
            "email_verified": True,
        }

    monkeypatch.setattr(
        backend_app,
        "_fetch_google_userinfo_from_access_token",
        fake_fetch_userinfo,
    )
    monkeypatch.setattr(
        backend_app,
        "generate_user_token_data",
        lambda user, db: {"sub": user.email, "user_type": user.user_type.value},
    )
    monkeypatch.setattr(
        backend_app,
        "create_access_token",
        lambda *, data, expires_delta: "signed-token",
    )

    result = asyncio.run(
        backend_app.google_auth(
            backend_app.GoogleAuthRequest(access_token="mobile-access-token"),
            db=object(),
        )
    )

    assert result == {"access_token": "signed-token", "token_type": "bearer"}
    assert fake_service.lookup_email == "new-google-user@example.com"
    assert fake_service.created_input.email == "new-google-user@example.com"
    assert fake_service.created_input.full_name == "Google User"
    assert fake_service.created_input.user_type == UserTypeEnum.owner
    assert captured["created_team_user_id"] == fake_service.created_user.id
    assert captured["created_team_input"].name == "Google User's team"


def test_register_user_uses_existing_pending_account_to_resend_code(monkeypatch):
    pending_user = _pending_owner_user()
    pending_user.email = "jane_doe@example.com"
    pending_user.full_name = None
    captured = {}

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def get_user_by_email(self, email):
            return pending_user

        def is_email_verification_pending(self, email):
            captured["pending_lookup_email"] = email
            return True

        def refresh_email_verification_code(self, email, *, verification_code, verification_expires_at):
            captured["refreshed_email"] = email
            captured["verification_code"] = verification_code
            captured["verification_expires_at"] = verification_expires_at
            return True

    monkeypatch.setattr(backend_app, "UserService", FakeUserService)
    monkeypatch.setattr(
        backend_app,
        "generate_email_verification_code",
        lambda: "112233",
    )
    monkeypatch.setattr(
        backend_app,
        "send_email_verification_email",
        lambda email, code, full_name=None: captured.update(
            {
                "sent_email": email,
                "sent_code": code,
                "sent_full_name": full_name,
            }
        ),
    )

    result = asyncio.run(
        backend_app.register_user(
            UserCreate(
                email="jane_doe@example.com",
                password="secret123",
                full_name=None,
            ),
            db=object(),
        )
    )

    assert result["status"] == "pending_verification"
    assert result["email"] == "jane_doe@example.com"
    assert captured["refreshed_email"] == "jane_doe@example.com"
    assert captured["verification_code"] == "112233"
    assert captured["sent_email"] == "jane_doe@example.com"
    assert captured["sent_code"] == "112233"


def test_google_auth_rejects_unconfigured_audience(monkeypatch):
    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def get_user_by_email(self, email):
            return None

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "legacy-client-id")
    monkeypatch.delenv("GOOGLE_WEB_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_SERVER_CLIENT_ID", raising=False)
    monkeypatch.setattr(backend_app, "UserService", FakeUserService)
    monkeypatch.setattr(
        backend_app.id_token,
        "verify_oauth2_token",
        lambda token, request, audience=None: {
            "aud": "some-other-client-id",
            "email": "wrong@example.com",
            "name": "Wrong User",
        },
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            backend_app.google_auth(
                backend_app.GoogleAuthRequest(id_token="fake-id-token"),
                db=object(),
            )
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid Google token"


def test_google_auth_uses_mobile_access_token(monkeypatch):
    created_user = _owner_user()
    captured = {}

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def get_user_by_email(self, email):
            captured["email_lookup"] = email
            return created_user

        def is_email_verified(self, user_id):
            captured["checked_verified_user_id"] = user_id
            return True

    async def fake_fetch_userinfo(access_token):
        captured["access_token"] = access_token
        return {
            "sub": "google-sub",
            "email": created_user.email,
            "name": "Owner",
            "email_verified": True,
        }

    monkeypatch.setattr(backend_app, "UserService", FakeUserService)
    monkeypatch.setattr(
        backend_app,
        "_fetch_google_userinfo_from_access_token",
        fake_fetch_userinfo,
    )
    monkeypatch.setattr(
        backend_app,
        "generate_user_token_data",
        lambda user, db: {"sub": user.email, "user_type": user.user_type.value},
    )
    monkeypatch.setattr(
        backend_app,
        "create_access_token",
        lambda *, data, expires_delta: "signed-token",
    )

    result = asyncio.run(
        backend_app.google_auth(
            backend_app.GoogleAuthRequest(access_token="mobile-access-token"),
            db=object(),
        )
    )

    assert result == {"access_token": "signed-token", "token_type": "bearer"}
    assert captured["access_token"] == "mobile-access-token"
    assert captured["email_lookup"] == created_user.email
    assert captured["checked_verified_user_id"] == created_user.id


def test_login_rejects_unverified_email_account(monkeypatch):
    pending_user = _pending_owner_user()

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def get_user_by_email(self, email):
            return pending_user

        def verify_password(self, plain_password, hashed_password):
            return True

        def is_email_verification_pending(self, email):
            return True

    monkeypatch.setattr(backend_app, "UserService", FakeUserService)

    form_data = type("FormData", (), {"username": "owner@example.com", "password": "secret123"})()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(backend_app.login_for_access_token(form_data=form_data, db=object()))

    assert exc_info.value.status_code == 403
    assert "Email not verified" in exc_info.value.detail


def test_staff_can_login_with_email_and_password(monkeypatch):
    staff_user = _staff_user()
    captured = {}

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def get_user_by_email(self, email):
            captured["email_lookup"] = email
            return staff_user

        def verify_password(self, plain_password, hashed_password):
            captured["password_attempt"] = plain_password
            captured["hashed_password"] = hashed_password
            return True

    expected_token_data = {
        "sub": staff_user.email,
        "user_type": staff_user.user_type.value,
        "staff_id": "staff-1",
        "team_id": "team-1",
        "permission_level": "view_only",
        "role": "assistant_coach",
        "app_role": "assistant_coach",
        "app_permissions": ["football.read"],
    }

    monkeypatch.setattr(backend_app, "UserService", FakeUserService)
    monkeypatch.setattr(
        backend_app,
        "generate_user_token_data",
        lambda user, db: expected_token_data,
    )
    monkeypatch.setattr(
        backend_app,
        "create_access_token",
        lambda *, data, expires_delta: "signed-token",
    )

    form_data = type("FormData", (), {"username": "Coach@Example.com", "password": "secret123"})()
    result = asyncio.run(backend_app.login_for_access_token(form_data=form_data, db=object()))

    assert result == {"access_token": "signed-token", "token_type": "bearer"}
    assert captured["email_lookup"] == "Coach@Example.com"
    assert captured["password_attempt"] == "secret123"
    assert captured["hashed_password"] == str(staff_user.password_hash)


def test_existing_staff_can_login_with_google(monkeypatch):
    staff_user = _staff_user()
    captured = {}

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def get_user_by_email(self, email):
            captured["email_lookup"] = email
            return staff_user

        def is_email_verified(self, user_id):
            captured["checked_verified_user_id"] = user_id
            return True

    async def fake_fetch_userinfo(access_token):
        captured["access_token"] = access_token
        return {
            "sub": "google-sub",
            "email": "Coach@Example.com",
            "name": "Coach User",
            "email_verified": True,
        }

    monkeypatch.setattr(backend_app, "UserService", FakeUserService)
    monkeypatch.setattr(
        backend_app,
        "_fetch_google_userinfo_from_access_token",
        fake_fetch_userinfo,
    )
    monkeypatch.setattr(
        backend_app,
        "generate_user_token_data",
        lambda user, db: {"sub": user.email, "user_type": user.user_type.value},
    )
    monkeypatch.setattr(
        backend_app,
        "create_access_token",
        lambda *, data, expires_delta: "signed-token",
    )

    result = asyncio.run(
        backend_app.google_auth(
            backend_app.GoogleAuthRequest(access_token="mobile-access-token"),
            db=object(),
        )
    )

    assert result == {"access_token": "signed-token", "token_type": "bearer"}
    assert captured["access_token"] == "mobile-access-token"
    assert captured["email_lookup"] == "Coach@Example.com"
    assert captured["checked_verified_user_id"] == staff_user.id


def test_user_read_model_excludes_password_hash():
    user = _owner_user()
    public_user = UserRead.model_validate(user.model_dump())

    dumped = public_user.model_dump()

    assert dumped["email"] == user.email
    assert dumped["user_type"] == UserTypeEnum.owner
    assert "password_hash" not in dumped
