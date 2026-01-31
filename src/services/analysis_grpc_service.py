import grpc
import os
import logging
from typing import AsyncGenerator, Optional

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
            # Use grpc.aio for async support
            self.channel = grpc.aio.insecure_channel(target)
            self.stub = analysis_pb2_grpc.AnalysisEngineStub(self.channel)
        return self.stub

    async def analyze_video(
        self, 
        video_path: str, 
        match_id: str, 
        calibration_path: str = "", 
        model_path: str = "",
        confidence_threshold: float = 0.5
    ) -> AsyncGenerator[analysis_pb2.VideoResponse, None]:
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
            async for response in responses:
                yield response
        except grpc.RpcError as e:
            logger.error(f"gRPC error during analysis: {e.details()}")
            yield analysis_pb2.VideoResponse(
                job_id=match_id,
                status="FAILED",
                message=f"gRPC Error: {e.details()}"
            )

    async def stream_analysis(
        self,
        chunks_generator: AsyncGenerator[analysis_pb2.VideoChunk, None]
    ) -> AsyncGenerator[analysis_pb2.MetricsUpdate, None]:
        """
        Sends a stream of video chunks to the gRPC server and yields metrics updates.
        """
        stub = self._get_stub()
        try:
            logger.info("Starting bi-directional streaming gRPC analysis (Async)")
            responses = stub.StreamAnalysis(chunks_generator)
            async for response in responses:
                yield response
        except grpc.RpcError as e:
            logger.error(f"gRPC error during streaming: {e.details()}")
            yield analysis_pb2.MetricsUpdate(
                status="FAILED",
                message=f"gRPC Error: {e.details()}"
            )

    async def close(self):
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
