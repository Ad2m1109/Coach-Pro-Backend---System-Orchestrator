from typing import List
from models.reunion import Reunion, ReunionCreate

class ReunionService:
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def _resolve_team_id(self, requested_team_id: str | None, user_team_ids: List[str]) -> str:
        if not user_team_ids:
            raise ValueError("Current user is not linked to any team.")
        if requested_team_id:
            if requested_team_id not in user_team_ids:
                raise ValueError("Team not accessible to the current user.")
            return requested_team_id
        if len(user_team_ids) == 1:
            return user_team_ids[0]
        raise ValueError("team_id is required when managing multiple teams.")

    def get_all_reunions(self, user_team_ids: List[str]) -> List[Reunion]:
        if not user_team_ids:
            return []
        with self.db_connection.cursor() as cursor:
            sql = """
                SELECT id, team_id, title, date, location, icon_name
                FROM reunions
                WHERE team_id IN %s
                ORDER BY date DESC, created_at DESC
            """
            cursor.execute(sql, (user_team_ids,))
            reunions_data = cursor.fetchall()
            return [Reunion(**r) for r in reunions_data]

    def create_reunion(self, reunion: ReunionCreate, user_team_ids: List[str]) -> Reunion:
        team_id = self._resolve_team_id(reunion.team_id, user_team_ids)
        with self.db_connection.cursor() as cursor:
            sql = "INSERT INTO reunions (id, team_id, title, date, location, icon_name) VALUES (UUID(), %s, %s, %s, %s, %s)"
            cursor.execute(
                sql,
                (team_id, reunion.title, reunion.date, reunion.location, reunion.icon_name),
            )
            self.db_connection.commit()
            cursor.execute(
                "SELECT * FROM reunions WHERE title = %s AND team_id = %s ORDER BY created_at DESC LIMIT 1",
                (reunion.title, team_id),
            )
            new_reunion = cursor.fetchone()
            return Reunion(**new_reunion)

    def delete_reunion(self, reunion_id: str, user_team_ids: List[str]) -> bool:
        if not user_team_ids:
            return False
        with self.db_connection.cursor() as cursor:
            sql = "DELETE FROM reunions WHERE id = %s AND team_id IN %s"
            cursor.execute(sql, (reunion_id, user_team_ids))
            self.db_connection.commit()
            return cursor.rowcount > 0
