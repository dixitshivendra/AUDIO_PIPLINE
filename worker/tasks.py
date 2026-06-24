"""Celery worker tasks for the RAKSHA audio pipeline.

Handles audio processing, MinIO archival, and webhook delivery.
Includes SSRF protection for webhook URLs and circuit breaker awareness.
"""

import os
import time
import json
import hmac
import hashlib
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from celery.signals import worker_process_init
from prometheus_client import start_http_server

from worker.celery_app import celery_app
from worker.llm_extraction import process_dispatch, process_general
from db.database import SessionLocal
from db.crud import save_job_result, save_quarantine_record, save_dead_letter
from core.metrics import (
    JOB_DURATION, JOB_SUCCESS, JOB_QUARANTINE, JOB_FAILED, JOB_DEAD_LETTER,
    get_json_logger,
)
from storage.client import upload_to_s3, download_from_s3, ensure_bucket_exists

logger = get_json_logger("celery_worker")

MAX_RETRIES = 3
METRICS_PORT = int(os.getenv("WORKER_METRICS_PORT", "8001"))

# SSRF protection: block private/internal IP ranges
_BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0",
    "169.254.169.254",  # AWS metadata
    "metadata.google.internal",  # GCP metadata
    "metadata.azure.com",  # Azure metadata
}


def _is_safe_url(url: str) -> bool:
    """Validate a webhook URL to prevent SSRF attacks.

    Blocks private IPs, localhost, cloud metadata endpoints,
    and non-HTTP(S) schemes.

    Args:
        url: The URL to validate.

    Returns:
        True if the URL is safe to request, False otherwise.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname or ""
    if hostname in _BLOCKED_HOSTS:
        return False

    # Block private IP ranges (10.x, 172.16-31.x, 192.168.x)
    if hostname.startswith("10.") or hostname.startswith("192.168."):
        return False
    if hostname.startswith("172."):
        try:
            second = int(hostname.split(".")[1])
            if 16 <= second <= 31:
                return False
        except (IndexError, ValueError):
            pass

    return True


@worker_process_init.connect
def init_worker(**kwargs: object) -> None:
    """Initialize worker metrics server and MinIO bucket on startup."""
    try:
        start_http_server(METRICS_PORT)
        logger.info("Worker metrics server started", extra={"port": METRICS_PORT})
    except OSError as e:
        logger.warning("Metrics server port in use", extra={"port": METRICS_PORT, "error": str(e)})
    try:
        ensure_bucket_exists()
        logger.info("MinIO bucket ensured on worker startup")
    except Exception as e:
        logger.warning("Failed to ensure bucket", extra={"error": str(e)})


def _send_webhook(job_id: str, result: dict, webhook_config: dict | None) -> None:
    """Send a webhook callback if configured.

    Includes SSRF protection, HMAC signing, and timeout enforcement.

    Args:
        job_id: The job ID for logging.
        result: The processing result dict.
        webhook_config: Dict with "url" and optional "secret" keys.
    """
    if not webhook_config or not webhook_config.get("url"):
        return

    url = webhook_config["url"]
    if not _is_safe_url(url):
        logger.error("Webhook URL blocked by SSRF protection", extra={"job_id": job_id, "url": url})
        return

    try:
        payload = json.dumps({
            "job_id": job_id,
            "status": result.get("status"),
            "extracted_data": result.get("extracted_data"),
            "error": result.get("reason"),
        }).encode()
        req = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if webhook_config.get("secret"):
            sig = hmac.new(
                webhook_config["secret"].encode(), payload, hashlib.sha256,
            ).hexdigest()
            req.add_header("X-Webhook-Signature", sig)
        urlopen(req, timeout=10)
        logger.info("Webhook sent", extra={"job_id": job_id, "url": url})
    except Exception as e:
        logger.error("Webhook failed", extra={"job_id": job_id, "error": str(e)})


def _check_alerts(db, job_id: str, org_id: str | None, extracted_data: dict) -> None:
    """Evaluate alert rules against completed job data and fire alerts.

    Checks:
    - sentiment_alert: triggers if sentiment_score meets threshold
    - keyword_alert: triggers if any configured keyword appears in transcript/keywords
    - compliance_alert: triggers if compliance_flags are present
    - competitor_alert: triggers if competitor_mentions are present
    """
    if not org_id:
        return

    from db.models import AlertRule, Alert
    import uuid

    rules = db.query(AlertRule).filter(
        AlertRule.org_id == org_id,
        AlertRule.is_active == True,
    ).all()

    for rule in rules:
        config = rule.config or {}
        rule_type = rule.rule_type
        fired = False
        title = ""
        message = ""
        matched_text = ""

        if rule_type == "sentiment_alert":
            threshold = config.get("threshold", 0.3)
            direction = config.get("direction", "negative")
            score = extracted_data.get("sentiment_score")
            if score is not None:
                if direction == "negative" and score <= threshold:
                    fired = True
                    title = f"Negative sentiment detected (score: {score:.2f})"
                    message = f"Sentiment score {score:.2f} fell below threshold {threshold}"
                elif direction == "positive" and score >= threshold:
                    fired = True
                    title = f"Positive sentiment detected (score: {score:.2f})"
                    message = f"Sentiment score {score:.2f} exceeded threshold {threshold}"

        elif rule_type == "keyword_alert":
            keywords = config.get("keywords", [])
            transcript_kw = extracted_data.get("keywords", [])
            found = [k for k in keywords if k.lower() in [w.lower() for w in transcript_kw]]
            if found:
                fired = True
                title = f"Keyword match: {', '.join(found[:3])}"
                message = f"Matched keywords: {', '.join(found)}"
                matched_text = ", ".join(found)

        elif rule_type == "compliance_alert":
            flags = extracted_data.get("compliance_flags", [])
            if flags:
                fired = True
                title = f"Compliance flag: {flags[0]}"
                message = f"{len(flags)} compliance concerns detected"
                matched_text = ", ".join(flags)

        elif rule_type == "competitor_alert":
            competitors = extracted_data.get("competitor_mentions", [])
            if competitors:
                fired = True
                title = f"Competitor mentioned: {', '.join(competitors)}"
                message = f"Competitor mentions detected in audio"
                matched_text = ", ".join(competitors)

        if fired:
            alert = Alert(
                id=str(uuid.uuid4()),
                org_id=org_id,
                rule_id=rule.id,
                job_id=job_id,
                severity=config.get("severity", "medium"),
                title=title,
                message=message,
                matched_text=matched_text,
            )
            db.add(alert)
            db.commit()
            logger.info("Alert fired", extra={"job_id": job_id, "rule_id": rule.id, "title": title})

            # Send webhook notifications for this alert
            try:
                _send_alert_webhooks(db, org_id, alert)
            except Exception as wh_err:
                logger.error("Alert webhook failed", extra={"job_id": job_id, "error": str(wh_err)})


def _send_alert_webhooks(db, org_id: str, alert) -> None:
    """Send webhook notifications for a triggered alert."""
    from db.models import Webhook

    webhooks = db.query(Webhook).filter(
        Webhook.org_id == org_id,
        Webhook.is_active == True,
    ).all()

    for wh in webhooks:
        if not _is_safe_url(wh.url):
            continue
        try:
            payload = json.dumps({
                "event": "alert_triggered",
                "alert_id": alert.id,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "job_id": alert.job_id,
            }).encode()
            req = Request(
                wh.url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            if wh.secret:
                sig = hmac.new(wh.secret.encode(), payload, hashlib.sha256).hexdigest()
                req.add_header("X-Webhook-Signature", sig)
            urlopen(req, timeout=10)
            wh.last_triggered_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            db.commit()
        except Exception:
            wh.failure_count += 1
            db.commit()


@celery_app.task(
    name="worker.tasks.check_alerts_task",
    max_retries=2,
    default_retry_delay=5,
    bind=True,
)
def check_alerts_task(
    self, job_id: str, org_id: str | None, extracted_data: dict,
) -> None:
    """独立的 Celery 任务：检查告警规则并触发告警。

    可以从 process_audio 中 fire-and-forget 调用，
    也可以由外部系统独立调度。
    """
    db = SessionLocal()
    try:
        _check_alerts(db, job_id, org_id, extracted_data)
    except Exception as e:
        logger.error("Alert check task failed", extra={"job_id": job_id, "error": str(e)})
        raise self.retry(exc=e)
    finally:
        db.close()


@celery_app.task(
    name="worker.tasks.process_audio",
    max_retries=3,
    default_retry_delay=10,
    bind=True,
)
def process_audio(
    self, job_id: str, s3_key: str, request_id: str | None = None,
    org_id: str | None = None,
) -> dict:
    """Process an audio file through the RAKSHA pipeline.

    Steps: download from S3 -> normalize -> transcribe -> extract -> persist.
    On permanent failure, moves to dead letter queue.

    Args:
        self: Celery task instance (for retry control).
        job_id: UUID string for this job.
        s3_key: S3 key of the uploaded audio file.
        request_id: Optional correlation ID from the API request.
        org_id: Optional organization ID for tenant-scoped storage.

    Returns:
        Dict with "status" and additional data.
    """
    start_time = time.time()
    logger.info("Started processing audio", extra={"job_id": job_id, "request_id": request_id})

    db = SessionLocal()
    local_file_path = None
    try:
        local_file_path = f"/tmp/{job_id}{os.path.splitext(s3_key)[1]}"
        download_from_s3(s3_key, local_file_path, org_id=org_id)
        logger.info("Downloaded audio from MinIO", extra={"job_id": job_id, "key": s3_key, "org_id": org_id})

        raw_audio_key = s3_key

        result = process_general(local_file_path)
        status = result.get("status", "unknown")

        clean_audio_key = None
        if status == "completed":
            clean_audio_key = f"clean/{job_id}.wav"
            try:
                with open(local_file_path, "rb") as f:
                    upload_to_s3(f, clean_audio_key, org_id=org_id)
                logger.info("Clean audio uploaded to MinIO", extra={"job_id": job_id, "key": clean_audio_key, "org_id": org_id})
            except Exception as e:
                logger.error("Failed to upload clean audio to MinIO", extra={"job_id": job_id, "error": str(e)})

        # Always save extracted data to the job record, even on quarantine.
        # This ensures library search works for quarantined jobs.
        extracted = result.get("extracted_data", {})
        transcript = result.get("transcript")
        if extracted or transcript:
            try:
                save_job_result(db, job_id, extracted, transcript=transcript)
            except Exception as save_err:
                logger.error("Failed to save job result", extra={"job_id": job_id, "error": str(save_err)})

        if status == "completed":
            JOB_SUCCESS.inc()
            logger.info("Job completed successfully", extra={"job_id": job_id})

            # Fire-and-forget: schedule alert check as independent task
            try:
                check_alerts_task.delay(job_id, org_id, extracted)
            except Exception as alert_err:
                logger.error("Failed to schedule alert check task", extra={"job_id": job_id, "error": str(alert_err)})
        else:
            save_quarantine_record(
                db,
                job_id,
                result.get("reason", "Unknown"),
                result.get("raw_transcript"),
                confidence_score=result.get("confidence"),
                raw_audio_key=raw_audio_key,
                clean_audio_key=clean_audio_key,
            )
            JOB_QUARANTINE.inc()
            logger.warning(
                "Job quarantined",
                extra={"job_id": job_id, "reason": result.get("reason")},
            )

        duration = time.time() - start_time
        JOB_DURATION.labels(status=status).observe(duration)
        logger.info(
            "Job duration recorded",
            extra={"job_id": job_id, "latency": duration, "status": status, "request_id": request_id},
        )

        result["raw_audio_key"] = raw_audio_key
        result["clean_audio_key"] = clean_audio_key
        return result

    except Exception as e:
        retries = self.request.retries
        if retries < MAX_RETRIES:
            logger.warning(
                "Job failed, retrying",
                extra={"job_id": job_id, "retry": retries + 1, "error": str(e)},
            )
            raise self.retry(exc=e)

        JOB_FAILED.inc()
        logger.error("Job permanently failed, moving to dead letter queue", extra={"job_id": job_id, "error": str(e)})
        try:
            save_dead_letter(db, job_id, f"MaxRetriesExceeded: {str(e)}", error_detail=str(e))
            JOB_DEAD_LETTER.inc()
        except Exception as dl_err:
            logger.error("Failed to save dead letter record", extra={"job_id": job_id, "error": str(dl_err)})

        return {
            "status": "dead_letter",
            "reason": f"MaxRetriesExceeded: {str(e)}",
        }
    finally:
        db.close()
        if os.path.exists(local_file_path):
            try:
                os.remove(local_file_path)
            except OSError:
                pass
