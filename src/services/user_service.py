from datetime import datetime
from typing import List, Optional
from models.user import User, UserCreate
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserService:
    _verification_schema_checked = False

    def __init__(self, db_connection):
        self.db_connection = db_connection

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.strip().lower()

    def _safe_rollback(self):
        rollback = getattr(self.db_connection, "rollback", None)
        if callable(rollback):
            rollback()

    def _ensure_verification_schema(self):
        if UserService._verification_schema_checked:
            return

        alterations = [
            "ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT TRUE",
            "ALTER TABLE users ADD COLUMN email_verification_code VARCHAR(12) NULL",
            "ALTER TABLE users ADD COLUMN email_verification_expires_at TIMESTAMP NULL",
            "ALTER TABLE users ADD COLUMN password_reset_code VARCHAR(12) NULL",
            "ALTER TABLE users ADD COLUMN password_reset_expires_at TIMESTAMP NULL",
        ]

        with self.db_connection.cursor() as cursor:
            for sql in alterations:
                try:
                    cursor.execute(sql)
                    self.db_connection.commit()
                except Exception:
                    self._safe_rollback()

            try:
                cursor.execute(
                    """
                    UPDATE users
                    SET email_verified = TRUE
                    WHERE email_verified IS NULL
                    """
                )
                self.db_connection.commit()
            except Exception:
                self._safe_rollback()

        UserService._verification_schema_checked = True

    def get_password_hash(self, password):
        return pwd_context.hash(password)

    def verify_password(self, plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    def create_user(self, user: UserCreate) -> User:
        self._ensure_verification_schema()
        normalized_email = self._normalize_email(user.email)
        if user.password:
            hashed_password = self.get_password_hash(user.password)
        else:
            import secrets
            hashed_password = self.get_password_hash(secrets.token_urlsafe(32))

        user_type = user.user_type.value if hasattr(user.user_type, "value") else str(user.user_type)
            
        with self.db_connection.cursor() as cursor:
            sql = """
                INSERT INTO users (
                    id, email, password_hash, full_name, user_type, app_role, is_active, email_verified
                )
                VALUES (UUID(), %s, %s, %s, %s, NULL, %s, TRUE)
            """
            cursor.execute(
                sql,
                (normalized_email, hashed_password, user.full_name, user_type, user.is_active),
            )
            self.db_connection.commit()
            cursor.execute("SELECT * FROM users WHERE email = %s ORDER BY created_at DESC LIMIT 1", (normalized_email,))
            new_user = cursor.fetchone()
            return User(**new_user)

    def create_pending_user(
        self,
        user: UserCreate,
        *,
        verification_code: str,
        verification_expires_at: datetime,
    ) -> User:
        self._ensure_verification_schema()
        normalized_email = self._normalize_email(user.email)
        hashed_password = self.get_password_hash(user.password or "")
        user_type = user.user_type.value if hasattr(user.user_type, "value") else str(user.user_type)

        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (
                    id,
                    email,
                    password_hash,
                    full_name,
                    user_type,
                    app_role,
                    is_active,
                    email_verified,
                    email_verification_code,
                    email_verification_expires_at
                )
                VALUES (UUID(), %s, %s, %s, %s, NULL, FALSE, FALSE, %s, %s)
                """,
                (
                    normalized_email,
                    hashed_password,
                    user.full_name,
                    user_type,
                    verification_code,
                    verification_expires_at,
                ),
            )
            self.db_connection.commit()
            cursor.execute(
                "SELECT * FROM users WHERE email = %s ORDER BY created_at DESC LIMIT 1",
                (normalized_email,),
            )
            new_user = cursor.fetchone()
            return User(**new_user)

    def get_user(self, user_id: str) -> Optional[User]:
        with self.db_connection.cursor() as cursor:
            sql = "SELECT * FROM users WHERE id = %s"
            cursor.execute(sql, (user_id,))
            user = cursor.fetchone()
            if user:
                return User(**user)
            return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        normalized_email = self._normalize_email(email)
        with self.db_connection.cursor() as cursor:
            sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(sql, (normalized_email,))
            user = cursor.fetchone()
            if user:
                return User(**user)
            return None

    def get_all_users(self) -> List[User]:
        with self.db_connection.cursor() as cursor:
            sql = "SELECT * FROM users"
            cursor.execute(sql)
            users = cursor.fetchall()
            return [User(**user) for user in users]

    def count_users(self) -> int:
        with self.db_connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM users")
            row = cursor.fetchone()
            return int(row["total"]) if row and "total" in row else 0

    def is_email_verification_pending(self, email: str) -> bool:
        self._ensure_verification_schema()
        normalized_email = self._normalize_email(email)
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT email_verified, email_verification_code, email_verification_expires_at
                FROM users
                WHERE email = %s
                """,
                (normalized_email,),
            )
            row = cursor.fetchone()
            if not row:
                return False
            return (
                not bool(row.get("email_verified", True))
                and bool(row.get("email_verification_code"))
            )

    def refresh_email_verification_code(
        self,
        email: str,
        *,
        verification_code: str,
        verification_expires_at: datetime,
    ) -> bool:
        self._ensure_verification_schema()
        normalized_email = self._normalize_email(email)
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET email_verification_code = %s,
                    email_verification_expires_at = %s
                WHERE email = %s
                  AND email_verified = FALSE
                """,
                (verification_code, verification_expires_at, normalized_email),
            )
            self.db_connection.commit()
            return cursor.rowcount > 0

    def verify_email_code(self, email: str, code: str) -> Optional[User]:
        self._ensure_verification_schema()
        normalized_email = self._normalize_email(email)
        normalized_code = code.strip()
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, email_verified, email_verification_code, email_verification_expires_at
                FROM users
                WHERE email = %s
                """,
                (normalized_email,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            if bool(row.get("email_verified", True)):
                return self.get_user(row["id"])
            if row.get("email_verification_code") != normalized_code:
                return None
            expires_at = row.get("email_verification_expires_at")
            if expires_at is None or expires_at < datetime.utcnow():
                return None
            return self.get_user(row["id"])

    def activate_verified_user(self, user_id: str) -> Optional[User]:
        self._ensure_verification_schema()
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET is_active = TRUE,
                    email_verified = TRUE,
                    email_verification_code = NULL,
                    email_verification_expires_at = NULL
                WHERE id = %s
                """,
                (user_id,),
            )
            self.db_connection.commit()
        return self.get_user(user_id)

    def is_email_verified(self, user_id: str) -> bool:
        self._ensure_verification_schema()
        with self.db_connection.cursor() as cursor:
            cursor.execute(
                "SELECT email_verified FROM users WHERE id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False
            return bool(row.get("email_verified", True))

    def start_password_reset(
        self,
        email: str,
        *,
        reset_code: str,
        reset_expires_at: datetime,
    ) -> Optional[User]:
        self._ensure_verification_schema()
        normalized_email = self._normalize_email(email)
        user = self.get_user_by_email(normalized_email)
        if not user or not user.is_active or not self.is_email_verified(user.id):
            return None

        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET password_reset_code = %s,
                    password_reset_expires_at = %s
                WHERE email = %s
                """,
                (reset_code, reset_expires_at, normalized_email),
            )
            self.db_connection.commit()
        return self.get_user_by_email(normalized_email)

    def reset_password_with_code(
        self,
        email: str,
        *,
        reset_code: str,
        new_password: str,
    ) -> Optional[User]:
        self._ensure_verification_schema()
        normalized_email = self._normalize_email(email)
        user = self.get_user_by_email(normalized_email)
        if not user:
            return None

        with self.db_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, password_reset_code, password_reset_expires_at
                FROM users
                WHERE email = %s
                """,
                (normalized_email,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            if row.get("password_reset_code") != reset_code.strip():
                return None
            expires_at = row.get("password_reset_expires_at")
            if expires_at is None or expires_at < datetime.utcnow():
                return None

            cursor.execute(
                """
                UPDATE users
                SET password_hash = %s,
                    password_reset_code = NULL,
                    password_reset_expires_at = NULL
                WHERE id = %s
                """,
                (self.get_password_hash(new_password), row["id"]),
            )
            self.db_connection.commit()
        return self.get_user(row["id"])

    def update_user(self, user_id: str, user_update: UserCreate) -> Optional[User]:
        existing_user = self.get_user(user_id)
        if not existing_user:
            return None

        self._ensure_verification_schema()
        normalized_email = self._normalize_email(user_update.email)
        hashed_password = (
            self.get_password_hash(user_update.password)
            if user_update.password
            else existing_user.password_hash
        )
        user_type = (
            user_update.user_type.value
            if hasattr(user_update.user_type, "value")
            else str(user_update.user_type)
        )
        with self.db_connection.cursor() as cursor:
            sql = """
                UPDATE users
                SET email = %s,
                    password_hash = %s,
                    full_name = %s,
                    user_type = %s,
                    app_role = NULL,
                    is_active = %s,
                    email_verified = TRUE,
                    email_verification_code = NULL,
                    email_verification_expires_at = NULL,
                    password_reset_code = NULL,
                    password_reset_expires_at = NULL
                WHERE id = %s
            """
            cursor.execute(
                sql,
                (
                    normalized_email,
                    hashed_password,
                    user_update.full_name,
                    user_type,
                    user_update.is_active,
                    user_id,
                ),
            )
            self.db_connection.commit()
            return self.get_user(user_id)

    def delete_user(self, user_id: str) -> bool:
        with self.db_connection.cursor() as cursor:
            sql = "DELETE FROM users WHERE id = %s"
            cursor.execute(sql, (user_id,))
            self.db_connection.commit()
            return cursor.rowcount > 0
