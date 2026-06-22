import os
import time
from celery import Celery
from celery.signals import worker_process_init
from prometheus_client import start_http_server

from worker.llm_extraction import process_dispatch
from database.db import save_job_result, save_quarantine_record
from core.metrics import JOB_DURATION, JOB_SUCCESS, JOB_QUARANTINE, JOB_FAILED, get_json_logger

logger = get_json_logger("celery_worker")

@worker_process_init.connect
def init_worker(**kwargs):
    try:
        start_http_server(8001)
        logger.info("Worker metrics server started on port 8001")
    except OSError:
        pass

celery_app = Celery(
    "audio_worker",
    broker=os.getenv("REDIS_URL", "redis://redis:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://redis:6379/0")
)

@celery_app.task(name="worker.tasks.process_audio")
def process_audio(job_id: str, local_file_path: str):
    start_time = time.time()
    logger.info("Started processing audio", extra={"job_id": job_id})
    
    try:
        result = process_dispatch(local_file_path)
        status = result.get("status")
        
        if status == "completed":
            save_job_result(job_id, result["extracted_data"])
            JOB_SUCCESS.inc()
            logger.info("Job completed successfully", extra={"job_id": job_id})
        else:
            save_quarantine_record(job_id, result.get("reason"), result.get("raw_transcript"))
            JOB_QUARANTINE.inc()
            logger.warning("Job quarantined", extra={"job_id": job_id, "reason": result.get("reason")})
            
        duration = time.time() - start_time
        JOB_DURATION.observe(duration)
        logger.info("Job duration recorded", extra={"job_id": job_id, "latency": duration})
        
        return result
        
    except Exception as e:
        JOB_FAILED.inc()
        logger.error("Job crashed", extra={"job_id": job_id, "error": str(e)})
        raise e
