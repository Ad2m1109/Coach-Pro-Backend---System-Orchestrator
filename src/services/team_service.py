from typing import List, Optional
from models.team import Team, TeamCreate
from fastapi import UploadFile
import os
import shutil
import uuid

class TeamService:
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def create_team(self, team: TeamCreate, user_id: str) -> Team:
        with self.db_connection.cursor() as cursor:
            sql = "INSERT INTO teams (id, name, user_id, primary_color, secondary_color, logo_url) VALUES (UUID(), %s, %s, %s, %s, %s)"
            cursor.execute(sql, (team.name, user_id, team.primary_color, team.secondary_color, team.logo_url))
            self.db_connection.commit()
            cursor.execute("SELECT * FROM teams WHERE name = %s AND user_id = %s ORDER BY created_at DESC LIMIT 1", (team.name, user_id))
            new_team = cursor.fetchone()
            return Team(**new_team)

    def get_team(self, team_id: str, user_id: str) -> Optional[Team]:
        with self.db_connection.cursor() as cursor:
            sql = """
                SELECT t.* FROM teams t 
                WHERE t.id = %s AND (t.user_id = %s OR EXISTS (SELECT 1 FROM staff s WHERE s.team_id = t.id AND s.user_id = %s))
            """
            cursor.execute(sql, (team_id, user_id, user_id))
            team = cursor.fetchone()
            if team:
                return Team(**team)
            return None

    def get_team_by_name(self, name: str) -> Optional[Team]:
        with self.db_connection.cursor() as cursor:
            sql = "SELECT * FROM teams WHERE name = %s"
            cursor.execute(sql, (name,))
            team = cursor.fetchone()
            if team:
                return Team(**team)
            return None

    def get_all_teams(self, user_id: str) -> List[Team]:
        with self.db_connection.cursor() as cursor:
            sql = """
                SELECT t.* FROM teams t WHERE t.user_id = %s
                UNION
                SELECT t.* FROM teams t JOIN staff s ON t.id = s.team_id WHERE s.user_id = %s
            """
            cursor.execute(sql, (user_id, user_id))
            teams = cursor.fetchall()
            return [Team(**team) for team in teams]

    def update_team(self, team_id: str, team_update: TeamCreate, user_id: str) -> Optional[Team]:
        team = self.get_team(team_id, user_id)
        if not team:
            return None

        with self.db_connection.cursor() as cursor:
            # Ensure logo_url is handled correctly, not overwritten with None if not provided
            logo_url = team_update.logo_url if team_update.logo_url is not None else team.logo_url
            sql = "UPDATE teams SET name = %s, primary_color = %s, secondary_color = %s, logo_url = %s WHERE id = %s AND user_id = %s"
            cursor.execute(sql, (team_update.name, team_update.primary_color, team_update.secondary_color, logo_url, team_id, user_id))
            self.db_connection.commit()
            return self.get_team(team_id, user_id)

    def delete_team(self, team_id: str) -> bool:
        with self.db_connection.cursor() as cursor:
            sql = "DELETE FROM teams WHERE id = %s"
            cursor.execute(sql, (team_id,))
            self.db_connection.commit()
            return cursor.rowcount > 0

    def save_team_logo(self, team_id: str, file: UploadFile, user_id: str) -> Team:
        # Define the path to save the image
        upload_dir = "static/images/teams"
        os.makedirs(upload_dir, exist_ok=True)

        # Generate a unique filename to prevent conflicts
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{team_id}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Construct the URL to access the file
        logo_url = f"/static/images/teams/{unique_filename}"

        # Update the team's logo_url in the database
        with self.db_connection.cursor() as cursor:
            sql = "UPDATE teams SET logo_url = %s WHERE id = %s"
            cursor.execute(sql, (logo_url, team_id))
            self.db_connection.commit()

        # Return the updated team object
        return self.get_team(team_id, user_id)
