"""Service wrapper for VideoAnalyzer with async job management."""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analyzer import VideoAnalyzer, get_config, find_videos_to_analyze, AnalysisConfig
from backend.utils.progress_manager import ProgressManager
from backend.models.schemas import AnalysisRequest, AnalysisOptions


class AnalysisService:
    """Async wrapper for VideoAnalyzer with progress broadcasting."""

    def __init__(self, progress_manager: ProgressManager):
        self.progress_manager = progress_manager
        self.logger = logging.getLogger(__name__)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._active_jobs: Dict[str, Dict[str, Any]] = {}

    async def start_analysis(self, request: AnalysisRequest) -> str:
        """
        Start a video analysis job asynchronously.

        Args:
            request: Analysis request with paths and options

        Returns:
            Job ID for tracking progress
        """
        job_id = self.progress_manager.create_job("analysis")
        self._active_jobs[job_id] = {"status": "pending", "request": request}

        # Start analysis in background
        asyncio.create_task(self._run_analysis(job_id, request))

        return job_id

    async def _run_analysis(self, job_id: str, request: AnalysisRequest) -> None:
        """Run analysis job in background with progress updates."""
        try:
            self._active_jobs[job_id]["status"] = "running"

            # Find videos to analyze
            if request.video_paths:
                video_files = [
                    (path, str(Path(path).with_suffix(".json")))
                    for path in request.video_paths
                    if Path(path).exists()
                ]
            else:
                video_files = find_videos_to_analyze(request.output_dir)

            if not video_files:
                self._active_jobs[job_id]["status"] = "completed"
                self._active_jobs[job_id]["result"] = {"total": 0, "analyzed": 0}
                await self.progress_manager.job_completed(job_id, result={"total": 0, "analyzed": 0})
                return

            # Create config from request options
            opts = request.analysis_options
            config = get_config(
                preset=opts.thoroughness.value,
                sample_frames=opts.sample_frames,
                sample_percentage=opts.sample_percent,
                color_clusters=opts.color_clusters,
                motion_resolution=opts.motion_res,
                face_model=opts.face_model.value if opts.face_model else None,
                workers=opts.workers,
                enable_audio=not opts.skip_audio,
                scene_detection=opts.scene_detection,
                full_resolution=opts.full_resolution,
            )

            # Create analyzer
            analyzer = VideoAnalyzer(config, self.logger)

            # Broadcast job started
            await self.progress_manager.job_started(job_id, "analysis", len(video_files))

            # Track results
            results = {
                "total": len(video_files),
                "analyzed": 0,
                "failed": 0,
                "videos": [],
                "errors": []
            }

            # Analyze each video with progress tracking
            for i, (video_path, json_path) in enumerate(video_files):
                try:
                    # Broadcast start
                    await self.progress_manager.analysis_start(
                        job_id, video_path, i + 1, len(video_files)
                    )

                    # Run analysis in executor
                    loop = asyncio.get_event_loop()
                    analysis_result = await loop.run_in_executor(
                        self._executor,
                        analyzer.analyze_video,
                        video_path
                    )

                    if analysis_result and not analysis_result.errors:
                        # Update metadata JSON
                        await loop.run_in_executor(
                            self._executor,
                            analyzer.update_metadata_file,
                            json_path,
                            analysis_result
                        )

                        results["analyzed"] += 1
                        results["videos"].append({
                            "path": video_path,
                            "success": True,
                            "processing_time_ms": analysis_result.processing_time_ms
                        })

                        await self.progress_manager.analysis_complete(
                            job_id, video_path, True,
                            metrics={"processing_time_ms": analysis_result.processing_time_ms}
                        )
                    else:
                        results["failed"] += 1
                        error_msg = analysis_result.errors[0] if analysis_result and analysis_result.errors else "Unknown error"
                        results["errors"].append({"path": video_path, "error": error_msg})

                        await self.progress_manager.analysis_complete(
                            job_id, video_path, False
                        )

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({"path": video_path, "error": str(e)})
                    await self.progress_manager.analysis_complete(job_id, video_path, False)

                # Update progress
                await self.progress_manager.job_progress(
                    job_id,
                    completed=results["analyzed"] + results["failed"],
                    total=len(video_files),
                    current_task=f"Analyzed {i + 1}/{len(video_files)}",
                    failed=results["failed"]
                )

            # Job completed
            self._active_jobs[job_id]["status"] = "completed"
            self._active_jobs[job_id]["result"] = results
            await self.progress_manager.job_completed(job_id, result=results)

        except Exception as e:
            self.logger.error(f"Analysis job {job_id} failed: {e}")
            self._active_jobs[job_id]["status"] = "failed"
            self._active_jobs[job_id]["error"] = str(e)
            await self.progress_manager.job_failed(job_id, str(e))

    async def analyze_single(self, video_path: str, options: AnalysisOptions) -> Dict[str, Any]:
        """
        Analyze a single video synchronously.

        Args:
            video_path: Path to video file
            options: Analysis options

        Returns:
            Analysis result dictionary
        """
        config = get_config(
            preset=options.thoroughness.value,
            sample_frames=options.sample_frames,
            sample_percentage=options.sample_percent,
            color_clusters=options.color_clusters,
            motion_resolution=options.motion_res,
            face_model=options.face_model.value if options.face_model else None,
            workers=options.workers,
            enable_audio=not options.skip_audio,
            scene_detection=options.scene_detection,
            full_resolution=options.full_resolution,
        )

        analyzer = VideoAnalyzer(config, self.logger)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            analyzer.analyze_video,
            video_path
        )

        return result.to_dict() if result else {"error": "Analysis failed"}

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an analysis job."""
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            state = self.progress_manager.get_job_state(job_id)
            return {
                "job_id": job_id,
                "status": job.get("status", "unknown"),
                "result": job.get("result"),
                "error": job.get("error"),
                "progress": state.get("data", {}) if state else {}
            }
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running analysis job."""
        if job_id in self._active_jobs:
            self._active_jobs[job_id]["status"] = "cancelled"
            return True
        return False
