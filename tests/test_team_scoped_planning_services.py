import os
import sys
from datetime import datetime
from pathlib import Path

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from models.reunion import ReunionCreate
from models.training_session import TrainingSessionCreate
from services.reunion_service import ReunionService
from services.training_session_service import TrainingSessionService


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rowcount = connection.rowcount

    def execute(self, sql, params=None):
        self.connection.executed.append((sql, params))
        self.rowcount = self.connection.rowcount

    def fetchall(self):
        return self.connection.fetchall_result

    def fetchone(self):
        return self.connection.fetchone_result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, *, fetchall_result=None, fetchone_result=None, rowcount=1):
        self.fetchall_result = fetchall_result or []
        self.fetchone_result = fetchone_result
        self.rowcount = rowcount
        self.executed = []
        self.commit_calls = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commit_calls += 1


def test_reunion_service_filters_by_team_ids():
    connection = FakeConnection(
        fetchall_result=[
            {
                "id": "reunion-1",
                "team_id": "team-1",
                "title": "Review",
                "date": datetime(2026, 4, 1, 10, 0, 0),
                "location": "Room A",
                "icon_name": "event",
            }
        ]
    )

    result = ReunionService(connection).get_all_reunions(["team-1"])

    assert len(result) == 1
    assert result[0].team_id == "team-1"
    assert connection.executed[0][1] == (["team-1"],)


def test_reunion_service_create_uses_single_accessible_team():
    connection = FakeConnection(
        fetchone_result={
            "id": "reunion-1",
            "team_id": "team-1",
            "title": "Review",
            "date": datetime(2026, 4, 1, 10, 0, 0),
            "location": "Room A",
            "icon_name": "event",
        }
    )

    reunion = ReunionCreate(
        title="Review",
        date=datetime(2026, 4, 1, 10, 0, 0),
        location="Room A",
        icon_name="event",
    )
    created = ReunionService(connection).create_reunion(reunion, ["team-1"])

    assert created.team_id == "team-1"
    assert connection.executed[0][1][0] == "team-1"
    assert connection.commit_calls == 1


def test_training_session_service_requires_explicit_team_for_multi_team_user():
    connection = FakeConnection()
    session = TrainingSessionCreate(
        title="Pressing",
        date=datetime(2026, 4, 2, 9, 0, 0),
        focus="High Press",
        icon_name="shield",
    )

    with pytest.raises(ValueError, match="team_id is required"):
        TrainingSessionService(connection).create_training_session(
            session,
            ["team-1", "team-2"],
        )


def test_training_session_delete_is_team_scoped():
    connection = FakeConnection(rowcount=1)

    deleted = TrainingSessionService(connection).delete_training_session(
        "session-1",
        ["team-1"],
    )

    assert deleted is True
    assert connection.executed[0][1] == ("session-1", ["team-1"])
    assert connection.commit_calls == 1
