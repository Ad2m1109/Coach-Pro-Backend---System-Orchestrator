import grpc
import os
import logging
from typing import Generator, Optional

import analysis_pb2
import analysis_pb2_grpc

logger = logging.getLogger(__name__)

class AnalysisGrpcService:
    def __init__(self, host: str = None, port: int = None):
        self.host = host or os.environ.get('ANALYSIS_ENGINE_HOST', 'localhost')
        self.port = port or int(os.environ.get('ANALYSIS_ENGINE_PORT', 50051))
        self.channel = None
        self.stub = None

    def _get_stub(self):
        if not self.stub:
            target = f"{self.host}:{self.port}"
            self.channel = grpc.insecure_channel(target)
            self.stub = analysis_pb2_grpc.AnalysisEngineStub(self.channel)
        return self.stub

    def analyze_video(
        self, 
        video_path: str, 
        match_id: str, 
        calibration_path: str = "", 
        model_path: str = "",
        confidence_threshold: float = 0.5
    ) -> Generator[analysis_pb2.VideoResponse, None, None]:
        """
        Sends a video analysis request to the gRPC server and yields progress updates.
        """
        stub = self._get_stub()
        request = analysis_pb2.VideoRequest(
            video_path=video_path,
            match_id=match_id,
            calibration_path=calibration_path,
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        
        try:
            logger.info(f"Starting gRPC analysis for match {match_id} at {video_path}")
            responses = stub.AnalyzeVideo(request)
            for response in responses:
                yield response
        except grpc.RpcError as e:
            logger.error(f"gRPC error during analysis for match {match_id}: {e.details()}")
            # Yield a failure response if the connection fails
            yield analysis_pb2.VideoResponse(
                job_id=match_id,
                status="FAILED",
                message=f"gRPC Error: {e.details()}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during gRPC analysis for match {match_id}: {e}")
            yield analysis_pb2.VideoResponse(
                job_id=match_id,
                status="FAILED",
                message=f"Unexpected Error: {str(e)}"
            )

    def stream_analysis(
        self,
        chunks_generator: Generator[analysis_pb2.VideoChunk, None, None]
    ) -> Generator[analysis_pb2.MetricsUpdate, None, None]:
        """
        Sends a stream of video chunks to the gRPC server and yields metrics updates.
        """
        stub = self._get_stub()
        try:
            logger.info("Starting bi-directional streaming gRPC analysis")
            responses = stub.StreamAnalysis(chunks_generator)
            for response in responses:
                yield response
        except grpc.RpcError as e:
            logger.error(f"gRPC error during streaming: {e.details()}")
            yield analysis_pb2.MetricsUpdate(
                status="FAILED",
                message=f"gRPC Error: {e.details()}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {e}")
            yield analysis_pb2.MetricsUpdate(
                status="FAILED",
                message=f"Unexpected Error: {str(e)}"
            )

    def close(self):
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None
