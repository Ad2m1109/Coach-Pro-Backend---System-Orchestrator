from typing import List, Optional
from models.user import User, UserCreate
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserService:
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def get_password_hash(self, password):
        return pwd_context.hash(password)

    def verify_password(self, plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    def create_user(self, user: UserCreate) -> User:
        if user.password:
            hashed_password = self.get_password_hash(user.password)
        else:
            import secrets
            hashed_password = self.get_password_hash(secrets.token_urlsafe(32))

        user_type = user.user_type.value if hasattr(user.user_type, "value") else str(user.user_type)
            
        with self.db_connection.cursor() as cursor:
            sql = """
                INSERT INTO users (id, email, password_hash, full_name, user_type, app_role, is_active)
                VALUES (UUID(), %s, %s, %s, %s, NULL, %s)
            """
            cursor.execute(
                sql,
                (user.email, hashed_password, user.full_name, user_type, user.is_active),
            )
            self.db_connection.commit()
            cursor.execute("SELECT * FROM users WHERE email = %s ORDER BY created_at DESC LIMIT 1", (user.email,))
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
        with self.db_connection.cursor() as cursor:
            sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(sql, (email,))
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

    def update_user(self, user_id: str, user_update: UserCreate) -> Optional[User]:
        existing_user = self.get_user(user_id)
        if not existing_user:
            return None

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
                    is_active = %s
                WHERE id = %s
            """
            cursor.execute(
                sql,
                (
                    user_update.email,
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
