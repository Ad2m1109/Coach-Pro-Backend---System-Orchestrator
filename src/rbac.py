from typing import Dict, List, Optional


APP_ROLE_ACCOUNT_MANAGER = "account_manager"
APP_ROLE_COACH = "coach"
APP_ROLE_ASSISTANT_COACH = "assistant_coach"
APP_ROLE_ANALYST = "analyst"
APP_ROLE_PLAYER = "player"


ROLE_PERMISSIONS: Dict[str, List[str]] = {
    APP_ROLE_ACCOUNT_MANAGER: ["accounts.read", "accounts.create", "accounts.update", "accounts.delete"],
    APP_ROLE_COACH: ["football.read", "football.write", "analysis.run", "notes.read", "notes.write"],
    APP_ROLE_ASSISTANT_COACH: ["football.read", "football.write", "analysis.run", "notes.read"],
    APP_ROLE_ANALYST: ["football.read", "analysis.run", "notes.read", "notes.write"],
    APP_ROLE_PLAYER: ["football.read"],
}


def derive_app_role(user_type: str, staff_role: Optional[str]) -> str:
    if user_type == "owner":
        return APP_ROLE_ACCOUNT_MANAGER
    if staff_role == "head_coach":
        return APP_ROLE_COACH
    if staff_role == "assistant_coach":
        return APP_ROLE_ASSISTANT_COACH
    if staff_role == "analyst":
        return APP_ROLE_ANALYST
    return APP_ROLE_PLAYER


def permissions_for_role(app_role: str) -> List[str]:
    return ROLE_PERMISSIONS.get(app_role, ROLE_PERMISSIONS[APP_ROLE_PLAYER])
