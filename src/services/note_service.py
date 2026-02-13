from typing import List, Optional
from models.note import MatchNote, MatchNoteCreate
import uuid

class NoteService:
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def create_note(self, note: MatchNoteCreate) -> MatchNote:
        with self.db_connection.cursor() as cursor:
            sql = """
                INSERT INTO match_notes (id, match_id, user_id, content, note_type, video_timestamp)
                VALUES (UUID(), %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                note.match_id,
                note.user_id,
                note.content,
                note.note_type.value,
                note.video_timestamp
            ))
            self.db_connection.commit()
            
            # Fetch the created note to return it
            cursor.execute("SELECT * FROM match_notes WHERE match_id = %s ORDER BY created_at DESC LIMIT 1", (note.match_id,))
            new_note = cursor.fetchone()
            return MatchNote(**new_note)

    def get_match_notes(self, match_id: str) -> List[MatchNote]:
        with self.db_connection.cursor() as cursor:
            # Join with users and staff to get names and roles
            # If user is owner, role is 'Owner'
            # If user is staff, get role from staff table
            # Also handle legacy 'proposition' notes by treating them as 'tactical'
            sql = """
                SELECT 
                    n.id,
                    n.match_id,
                    n.user_id,
                    n.content,
                    CASE 
                        WHEN n.note_type = 'proposition' THEN 'tactical'
                        ELSE n.note_type
                    END as note_type,
                    n.video_timestamp,
                    n.created_at,
                    COALESCE(s.name, u.full_name) as author_name,
                    CASE 
                        WHEN u.user_type = 'owner' THEN 'Owner'
                        ELSE COALESCE(s.role, 'Staff')
                    END as author_role
                FROM match_notes n
                JOIN users u ON n.user_id = u.id
                LEFT JOIN staff s ON u.id = s.user_id
                WHERE n.match_id = %s 
                ORDER BY n.created_at ASC
            """
            cursor.execute(sql, (match_id,))
            notes = cursor.fetchall()
            return [MatchNote(**n) for n in notes]

    def delete_note(self, note_id: str, user_id: str) -> bool:
        """Only the author can delete their note"""
        with self.db_connection.cursor() as cursor:
            sql = "DELETE FROM match_notes WHERE id = %s AND user_id = %s"
            cursor.execute(sql, (note_id, user_id))
            self.db_connection.commit()
            return cursor.rowcount > 0
