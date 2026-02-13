"""
Service for managing video analysis jobs.
Routes jobs to the Tracking Engine gRPC service and tracks results.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from services.tracking_engine_client import TrackingEngineClient
from models.analysis_report import AnalysisReport

logger = logging.getLogger(__name__)


class AnalysisJobService:
    """Manages video analysis job lifecycle."""

    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.engine_client = TrackingEngineClient()

    def create_analysis_job(self, match_id: str, video_path: str, user_id: str) -> dict:
        """
        Create and submit an analysis job.
        
        Args:
            match_id: Match identifier
            video_path: Path to video file to analyze
            user_id: User who submitted the job
            
        Returns:
            dict with job details
        """
        job_id = str(uuid.uuid4())
        
        with self.db_connection.cursor() as cursor:
            # Create analysis report record
            sql = """
                INSERT INTO analysis_reports (id, match_id, report_type, report_data, generated_by, generated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            report_data = {
                'job_id': job_id,
                'status': 'PENDING',
                'progress': 0.0,
                'submitted_at': datetime.utcnow().isoformat(),
                'video_path': video_path,
                'outputs': {}
            }
            
            cursor.execute(sql, (
                job_id,
                match_id,
                'video_analysis',
                json.dumps(report_data),
                user_id,
                datetime.utcnow()
            ))
            self.db_connection.commit()
        
        logger.info(f"Created analysis job {job_id} for match {match_id}")
        
        return {
            'job_id': job_id,
            'match_id': match_id,
            'status': 'PENDING'
        }

    async def submit_analysis_job(
        self,
        job_id: str,
        match_id: str,
        video_path: str,
        frame_limit: int = 0,
        skip_json: bool = False
    ) -> dict:
        """
        Submit a job to the Tracking Engine and monitor progress.
        
        Args:
            job_id: Unique job identifier
            match_id: Match identifier
            video_path: Path to video file
            frame_limit: Max frames to process
            skip_json: Skip JSON export
            
        Returns:
            Final analysis results
        """
        
        async def progress_callback(response):
            """Update job progress in database"""
            await self._update_job_progress(job_id, response)
        
        try:
            # Submit to tracking engine
            result = await self.engine_client.analyze_video(
                video_path=video_path,
                match_id=match_id,
                frame_limit=frame_limit,
                skip_json=skip_json,
                progress_callback=progress_callback
            )
            
            # Store final results
            await self._save_analysis_results(job_id, match_id, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Analysis job {job_id} failed: {e}")
            await self._update_job_status(job_id, 'FAILED', str(e))
            raise

    async def _update_job_progress(self, job_id: str, response):
        """Update job progress in database"""
        try:
            with self.db_connection.cursor() as cursor:
                sql = """
                    UPDATE analysis_reports
                    SET report_data = JSON_SET(
                        report_data,
                        '$.status', %s,
                        '$.progress', %s,
                        '$.message', %s
                    )
                    WHERE id = %s
                """
                cursor.execute(sql, (
                    response.status,
                    float(response.progress),
                    response.message,
                    job_id
                ))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to update job progress: {e}")

    async def _update_job_status(self, job_id: str, status: str, message: str = ''):
        """Update job status"""
        try:
            with self.db_connection.cursor() as cursor:
                sql = """
                    UPDATE analysis_reports
                    SET report_data = JSON_SET(
                        report_data,
                        '$.status', %s,
                        '$.message', %s
                    )
                    WHERE id = %s
                """
                cursor.execute(sql, (status, message, job_id))
                self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

    async def _save_analysis_results(self, job_id: str, match_id: str, result: dict):
        """Save analysis results to database"""
        try:
            with self.db_connection.cursor() as cursor:
                report_data = {
                    'job_id': job_id,
                    'status': result.get('status'),
                    'message': result.get('message'),
                    'completed_at': datetime.utcnow().isoformat(),
                    'outputs': result.get('result', {})
                }
                
                sql = """
                    UPDATE analysis_reports
                    SET report_data = %s
                    WHERE id = %s
                """
                cursor.execute(sql, (json.dumps(report_data), job_id))
                self.db_connection.commit()
                
                logger.info(f"Saved results for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
            raise

    def get_job_status(self, job_id: str) -> dict:
        """Get current job status"""
        with self.db_connection.cursor() as cursor:
            sql = "SELECT report_data FROM analysis_reports WHERE id = %s"
            cursor.execute(sql, (job_id,))
            result = cursor.fetchone()
            
            if not result:
                return {'status': 'NOT_FOUND'}
            
            report_data = json.loads(result['report_data'])
            return {
                'job_id': job_id,
                'status': report_data.get('status'),
                'progress': report_data.get('progress', 0),
                'message': report_data.get('message', ''),
                'outputs': report_data.get('outputs', {})
            }

    def get_match_analysis(self, match_id: str) -> Optional[dict]:
        """Get analysis results for a match"""
        with self.db_connection.cursor() as cursor:
            sql = """
                SELECT id, report_data
                FROM analysis_reports
                WHERE match_id = %s
                ORDER BY generated_at DESC
                LIMIT 1
            """
            cursor.execute(sql, (match_id,))
            result = cursor.fetchone()
            
            if not result:
                return None
            
            raw_report = result['report_data']
            report_data = raw_report if isinstance(raw_report, dict) else json.loads(raw_report)
            return {
                'job_id': result['id'],
                'match_id': match_id,
                'status': report_data.get('status'),
                'progress': report_data.get('progress', 0),
                'message': report_data.get('message', ''),
                'outputs': report_data.get('outputs', {})
            }
