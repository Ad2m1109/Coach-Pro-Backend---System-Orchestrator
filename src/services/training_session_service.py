from typing import List
from models.training_session import TrainingSession, TrainingSessionCreate

class TrainingSessionService:
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

    def get_all_training_sessions(self, user_team_ids: List[str]) -> List[TrainingSession]:
        if not user_team_ids:
            return []
        with self.db_connection.cursor() as cursor:
            sql = """
                SELECT id, team_id, title, date, focus, icon_name
                FROM training_sessions
                WHERE team_id IN %s
                ORDER BY date DESC, created_at DESC
            """
            cursor.execute(sql, (user_team_ids,))
            sessions_data = cursor.fetchall()
            return [TrainingSession(**s) for s in sessions_data]

    def create_training_session(self, session: TrainingSessionCreate, user_team_ids: List[str]) -> TrainingSession:
        team_id = self._resolve_team_id(session.team_id, user_team_ids)
        with self.db_connection.cursor() as cursor:
            sql = "INSERT INTO training_sessions (id, team_id, title, date, focus, icon_name) VALUES (UUID(), %s, %s, %s, %s, %s)"
            cursor.execute(
                sql,
                (team_id, session.title, session.date, session.focus, session.icon_name),
            )
            self.db_connection.commit()
            cursor.execute(
                "SELECT * FROM training_sessions WHERE title = %s AND team_id = %s ORDER BY created_at DESC LIMIT 1",
                (session.title, team_id),
            )
            new_session = cursor.fetchone()
            return TrainingSession(**new_session)

    def delete_training_session(self, session_id: str, user_team_ids: List[str]) -> bool:
        if not user_team_ids:
            return False
        with self.db_connection.cursor() as cursor:
            sql = "DELETE FROM training_sessions WHERE id = %s AND team_id IN %s"
            cursor.execute(sql, (session_id, user_team_ids))
            self.db_connection.commit()
            return cursor.rowcount > 0
