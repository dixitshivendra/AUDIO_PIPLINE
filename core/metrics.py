import logging
from pythonjsonlogger import jsonlogger
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("api_requests_total", "Total API Requests")
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "API Request Duration")
UPLOAD_COUNT = Counter("audio_uploads_total", "Uploaded Audio Files")

JOB_DURATION = Histogram("audio_job_duration_seconds",
                         "Audio Processing Duration")
JOB_SUCCESS = Counter("jobs_completed_total", "Completed Jobs")
JOB_QUARANTINE = Counter("jobs_quarantine_total", "Quarantined Jobs")
JOB_FAILED = Counter("jobs_failed_total", "Failed Jobs")


def get_json_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
