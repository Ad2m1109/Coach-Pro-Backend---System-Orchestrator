from typing import List, Optional
from models.staff import Staff, StaffCreate, StaffCreateWithAccount, PermissionLevelEnum
from models.user import UserCreate, UserTypeEnum
from passlib.context import CryptContext
import uuid

class StaffService:
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def create_staff_with_account(self, staff_data: StaffCreateWithAccount, user_team_ids: List[str]) -> Staff:
        """Create a staff member with a new user account"""
        if staff_data.team_id not in user_team_ids:
            raise ValueError("Team not owned by current user.")
        
        with self.db_connection.cursor() as cursor:
            # Check if email already exists
            cursor.execute("SELECT id FROM users WHERE email = %s", (staff_data.email,))
            if cursor.fetchone():
                raise ValueError("Email already exists")
            
            # Create user account
            user_id = str(uuid.uuid4())
            password_hash = self.pwd_context.hash(staff_data.password)
            
            cursor.execute(
                "INSERT INTO users (id, email, password_hash, full_name, user_type, is_active) VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, staff_data.email, password_hash, staff_data.name, UserTypeEnum.staff.value, True)
            )
            
            # Create staff record
            staff_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO staff (id, team_id, user_id, name, role, permission_level, email) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (staff_id, staff_data.team_id, user_id, staff_data.name, staff_data.role.value, staff_data.permission_level.value, staff_data.email)
            )
            
            self.db_connection.commit()
            
            cursor.execute("SELECT * FROM staff WHERE id = %s", (staff_id,))
            new_staff = cursor.fetchone()
            return Staff(**new_staff)

    def create_staff(self, staff: StaffCreate, user_team_ids: List[str]) -> Staff:
        """Create staff without user account (legacy method)"""
        if staff.team_id not in user_team_ids:
            raise ValueError("Team not owned by current user.")
        with self.db_connection.cursor() as cursor:
            sql = "INSERT INTO staff (id, team_id, name, role, permission_level, email, user_id) VALUES (UUID(), %s, %s, %s, %s, %s, %s)"
            cursor.execute(sql, (
                staff.team_id, 
                staff.name, 
                staff.role.value if staff.role else None,
                staff.permission_level.value,
                staff.email,
                staff.user_id
            ))
            self.db_connection.commit()
            cursor.execute("SELECT * FROM staff WHERE name = %s ORDER BY id DESC LIMIT 1", (staff.name,))
            new_staff = cursor.fetchone()
            return Staff(**new_staff)

    def get_staff(self, staff_id: str, user_team_ids: List[str]) -> Optional[Staff]:
        with self.db_connection.cursor() as cursor:
            sql = "SELECT * FROM staff WHERE id = %s AND team_id IN %s"
            cursor.execute(sql, (staff_id, user_team_ids))
            staff = cursor.fetchone()
            if staff:
                return Staff(**staff)
            return None

    def get_all_staff(self, user_team_ids: List[str]) -> List[Staff]:
        if not user_team_ids:
            return []
        with self.db_connection.cursor() as cursor:
            sql = "SELECT * FROM staff WHERE team_id IN %s"
            cursor.execute(sql, (user_team_ids,))
            staff = cursor.fetchall()
            return [Staff(**s) for s in staff]

    def update_staff(self, staff_id: str, staff_update: StaffCreate, user_team_ids: List[str]) -> Optional[Staff]:
        if staff_update.team_id not in user_team_ids:
            raise ValueError("Team not owned by current user.")
        with self.db_connection.cursor() as cursor:
            sql = "UPDATE staff SET team_id = %s, name = %s, role = %s, permission_level = %s, email = %s WHERE id = %s AND team_id IN %s"
            cursor.execute(sql, (
                staff_update.team_id, 
                staff_update.name, 
                staff_update.role.value if staff_update.role else None, 
                staff_update.permission_level.value,
                staff_update.email,
                staff_id, 
                user_team_ids
            ))
            self.db_connection.commit()
            return self.get_staff(staff_id, user_team_ids)

    def delete_staff(self, staff_id: str, user_team_ids: List[str]) -> bool:
        with self.db_connection.cursor() as cursor:
            sql = "DELETE FROM staff WHERE id = %s AND team_id IN %s"
            cursor.execute(sql, (staff_id, user_team_ids))
            self.db_connection.commit()
            return cursor.rowcount > 0

    @staticmethod
    def check_permission(staff: Staff, required_permission: str) -> bool:
        """Check if staff has required permission"""
        permission_hierarchy = {
            'full_access': ['edit', 'view', 'notes'],
            'view_only': ['view'],
            'notes_only': ['notes']
        }
        allowed_permissions = permission_hierarchy.get(staff.permission_level, [])
        return required_permission in allowed_permissions

    def get_staff_by_user_id(self, user_id: str) -> Optional[Staff]:
        """Get staff record by user_id"""
        with self.db_connection.cursor() as cursor:
            cursor.execute("SELECT * FROM staff WHERE user_id = %s", (user_id,))
            staff = cursor.fetchone()
            if staff:
                return Staff(**staff)
            return None
