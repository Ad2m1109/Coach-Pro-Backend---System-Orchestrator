"""
gRPC client for the Tracking Engine service.
Used by the backend to submit video analysis jobs.
"""

import grpc
import logging
import os
from typing import Optional, Callable
from analysis_pb2 import AnalysisRequest
import analysis_pb2_grpc

logger = logging.getLogger(__name__)


class TrackingEngineClient:
    """Client for communicating with the Tracking Engine gRPC service."""

    def __init__(self, host: str = None, port: int = None):
        """
        Initialize the client.
        
        Args:
            host: gRPC server host (defaults to env var or localhost)
            port: gRPC server port (defaults to env var or 50051)
        """
        self.host = host or os.environ.get('TRACKING_ENGINE_HOST', 'localhost')
        self.port = port or int(os.environ.get('TRACKING_ENGINE_PORT', 50051))
        self.channel = None
        self.stub = None

    def connect(self):
        """Establish connection to the gRPC server."""
        try:
            address = f'{self.host}:{self.port}'
            self.channel = grpc.aio.insecure_channel(address)
            self.stub = analysis_pb2_grpc.AnalysisServiceStub(self.channel)
            logger.info(f"Connected to Tracking Engine at {address}")
        except Exception as e:
            logger.error(f"Failed to connect to Tracking Engine: {e}")
            raise

    async def analyze_video(
        self,
        video_path: str,
        match_id: str,
        frame_limit: int = 0,
        skip_json: bool = False,
        confidence_threshold: float = 0.5,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        Submit a video for analysis and stream progress updates.
        
        Args:
            video_path: Path to the video file
            match_id: Unique match identifier
            frame_limit: Max frames to process (0 = all)
            skip_json: Skip JSON export if True
            confidence_threshold: YOLO confidence threshold
            progress_callback: Optional callback for progress updates
            
        Returns:
            dict with final analysis results
        """
        if not self.stub:
            self.connect()
        
        request = AnalysisRequest(
            video_path=video_path,
            match_id=match_id,
            frame_limit=frame_limit,
            skip_json=skip_json,
            confidence_threshold=confidence_threshold
        )
        
        logger.info(f"Submitting analysis request for match {match_id}: {video_path}")
        
        try:
            response_stream = self.stub.AnalyzeVideo(request)
            
            final_response = None
            async for response in response_stream:
                logger.info(f"[{match_id}] {response.status}: {response.message} ({response.progress*100:.1f}%)")
                
                # Call progress callback if provided
                if progress_callback:
                    await progress_callback(response)
                
                # Store final response
                if response.status in ['COMPLETED', 'FAILED']:
                    final_response = response
            
            if not final_response:
                raise RuntimeError("No response from server")
            
            return self._response_to_dict(final_response)
        
        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise

    def analyze_video_sync(
        self,
        video_path: str,
        match_id: str,
        frame_limit: int = 0,
        skip_json: bool = False,
        confidence_threshold: float = 0.5,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        Synchronous wrapper for analyze_video (uses asyncio.run internally).
        """
        import asyncio
        return asyncio.run(self.analyze_video(
            video_path, match_id, frame_limit, skip_json, confidence_threshold, progress_callback
        ))

    def _response_to_dict(self, response) -> dict:
        """Convert protobuf response to dict"""
        result = {
            'match_id': response.match_id,
            'status': response.status,
            'message': response.message,
            'progress': response.progress,
        }
        
        if response.HasField('result'):
            result['result'] = {
                'match_id': response.result.match_id,
                'total_frames': response.result.total_frames,
                'players_tracked': response.result.players_tracked,
                'tracking_video_path': response.result.tracking_video_path,
                'tracking_json_path': response.result.tracking_json_path,
                'backline_video_path': response.result.backline_video_path,
                'heatmap_video_path': response.result.heatmap_video_path,
                'possession_analysis_path': response.result.possession_analysis_path,
                'animation_video_path': response.result.animation_video_path,
                'processing_time_seconds': response.result.processing_time_seconds,
                'fps': response.result.fps,
                'error_message': response.result.error_message,
            }
        
        return result

    async def close(self):
        """Close the async gRPC channel."""
        if self.channel:
            await self.channel.close()
            logger.info("Connection closed")
