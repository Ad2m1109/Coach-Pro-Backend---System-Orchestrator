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


def test_register_user_issues_full_claim_token(monkeypatch):
    created_user = _owner_user()
    captured = {}

    class FakeUserService:
        def __init__(self, db):
            self.db = db

        def count_users(self):
            return 0

        def get_user_by_email(self, email):
            return None

        def create_user(self, user_create):
            captured["created_user_input"] = user_create
            return created_user

    expected_token_data = {
        "sub": created_user.email,
        "user_type": created_user.user_type.value,
        "app_role": "account_manager",
        "app_permissions": ["accounts.read"],
    }

    monkeypatch.setattr(backend_app, "UserService", FakeUserService)
    monkeypatch.setattr(
        backend_app,
        "generate_user_token_data",
        lambda user, db: expected_token_data,
    )

    def fake_create_access_token(*, data, expires_delta):
        captured["token_data"] = data
        captured["expires_delta"] = expires_delta
        return "signed-token"

    monkeypatch.setattr(backend_app, "create_access_token", fake_create_access_token)

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

    assert result == {"access_token": "signed-token", "token_type": "bearer"}
    assert captured["token_data"] == expected_token_data
    assert captured["created_user_input"].email == "owner@example.com"


def test_google_auth_auto_provisions_user_when_email_not_found(monkeypatch):
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

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "legacy-client-id")
    monkeypatch.setenv("GOOGLE_WEB_CLIENT_ID", "web-client-id")
    monkeypatch.setenv("GOOGLE_SERVER_CLIENT_ID", "server-client-id")
    fake_service = FakeUserService(object())
    monkeypatch.setattr(backend_app, "UserService", lambda db: fake_service)
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


def test_user_read_model_excludes_password_hash():
    user = _owner_user()
    public_user = UserRead.model_validate(user.model_dump())

    dumped = public_user.model_dump()

    assert dumped["email"] == user.email
    assert dumped["user_type"] == UserTypeEnum.owner
    assert "password_hash" not in dumped
