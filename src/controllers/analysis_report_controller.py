from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import get_db, Connection
from services.analysis_report_service import AnalysisReportService
from models.analysis_report import AnalysisReport, AnalysisReportCreate
from dependencies import get_current_active_user # Import the dependency
from services.team_service import TeamService # New import
from models.user import User # Import User model

router = APIRouter()

@router.post("/analysis_reports", response_model=AnalysisReport)
def create_analysis_report(report: AnalysisReportCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = AnalysisReportService(db)
    return service.create_analysis_report(report, user_team_ids)

@router.get("/analysis_reports", response_model=List[AnalysisReport])
def get_all_analysis_reports(db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = AnalysisReportService(db)
    return service.get_all_analysis_reports(user_team_ids)

@router.get("/analysis_reports/{report_id}", response_model=AnalysisReport)
def get_analysis_report(report_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = AnalysisReportService(db)
    report = service.get_analysis_report(report_id, user_team_ids)
    if not report:
        raise HTTPException(status_code=404, detail="Analysis Report not found")
    return report

@router.put("/analysis_reports/{report_id}", response_model=AnalysisReport)
def update_analysis_report(report_id: str, report_update: AnalysisReportCreate, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = AnalysisReportService(db)
    report = service.update_analysis_report(report_id, report_update, user_team_ids)
    if not report:
        raise HTTPException(status_code=404, detail="Analysis Report not found")
    return report

@router.delete("/analysis_reports/{report_id}")
def delete_analysis_report(report_id: str, db: Connection = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    team_service = TeamService(db)
    user_teams = team_service.get_all_teams(current_user.id)
    user_team_ids = [team.id for team in user_teams]
    service = AnalysisReportService(db)
    if not service.delete_analysis_report(report_id, user_team_ids):
        raise HTTPException(status_code=404, detail="Analysis Report not found")
    return {"message": "Analysis Report deleted successfully"}
