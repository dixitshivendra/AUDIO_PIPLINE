"""Database CRUD operations for jobs, quarantine records, and dead letter queue.

Provides persistence layer for the RAKSHA audio pipeline including
job tracking, quarantine review workflow, dead letter queue, and
per-tenant usage metering.
All write operations use a single transaction per call to ensure atomicity.
"""

from sqlalchemy.orm import Session

from db.models import JobRecord, QuarantineRecord, DeadLetterRecord, Organization, UsageRecord


def save_job_result(db: Session, job_id: str, extracted_data: dict, transcript: str | None = None) -> None:
    """Update an existing job record with completed status and extracted data.

    Also populates searchable columns (summary, sentiment, keywords, etc.)
    and increments the organization's monthly usage counter.

    Args:
        db: Active SQLAlchemy session.
        job_id: UUID string identifying the job to update.
        extracted_data: Validated extraction dict from LLM.
        transcript: Full transcript text if available.

    Raises:
        ValueError: If no job with the given ID exists.
    """
    from datetime import datetime, timezone

    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise ValueError(f"Job {job_id} not found")
    job.status = "completed"
    job.extracted_data = extracted_data
    job.completed_at = datetime.now(timezone.utc)

    if transcript:
        job.transcript = transcript

    # Populate searchable columns from extracted_data
    if extracted_data:
        job.summary = extracted_data.get("summary")
        job.sentiment = extracted_data.get("sentiment")
        job.sentiment_score = extracted_data.get("sentiment_score")
        job.keywords = extracted_data.get("keywords")
        job.topics = extracted_data.get("topics")
        job.action_items = extracted_data.get("action_items")
        job.decisions = extracted_data.get("decisions")
        job.language = extracted_data.get("language")
        job.speakers = extracted_data.get("speakers")

        # Extract entities into flat fields
        entities = extracted_data.get("entities", {})
        if entities:
            extra = job.extra_metadata or {}
            extra["entities"] = entities
            job.extra_metadata = extra

        # Merge compliance flags and competitor mentions
        compliance = extracted_data.get("compliance_flags", [])
        competitors = extracted_data.get("competitor_mentions", [])
        if compliance or competitors:
            extra = job.extra_metadata or {}
            if compliance:
                extra["compliance_flags"] = compliance
            if competitors:
                extra["competitor_mentions"] = competitors
            job.extra_metadata = extra

    # Increment usage for the org
    if job.org_id:
        _increment_org_usage(db, job.org_id, completed=True)

    db.commit()


def save_quarantine_record(
    db: Session,
    job_id: str,
    reason: str,
    transcript_excerpt: str | None = None,
    confidence_score: float | None = None,
    raw_audio_key: str = "",
    clean_audio_key: str | None = None,
) -> None:
    """Create a quarantine record and mark the parent job as quarantined.

    Also increments the organization's monthly usage counter.

    Used when audio fails confidence validation or LLM extraction produces
    ambiguous results requiring human review. Both the job update and record
    creation happen in a single transaction.

    Args:
        db: Active SQLAlchemy session.
        job_id: UUID string identifying the parent job.
        reason: Human-readable explanation for quarantine (e.g., exception message).
        transcript_excerpt: First N characters of the raw transcript.
        confidence_score: Average log-probability from Whisper transcription.
        raw_audio_key: S3 object key for the original uploaded audio.
        clean_audio_key: S3 object key for the FFmpeg-processed audio (may be None).

    Raises:
        ValueError: If no job with the given ID exists.
    """
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise ValueError(f"Job {job_id} not found")

    job.status = "quarantine"
    job.error_detail = reason

    record = QuarantineRecord(
        job_id=job_id,
        reason=reason,
        transcript_excerpt=transcript_excerpt,
        confidence_score=confidence_score,
        raw_audio_key=raw_audio_key,
        clean_audio_key=clean_audio_key,
    )
    db.add(record)

    # Increment usage (quarantined jobs still count)
    if job.org_id:
        _increment_org_usage(db, job.org_id, completed=False)

    db.commit()


def save_dead_letter(
    db: Session,
    job_id: str,
    reason: str,
    raw_audio_key: str = "",
    error_detail: str | None = None,
) -> None:
    """Move a permanently failed job to the dead letter queue.

    Called after all Celery retries are exhausted. The job retains its audio
    key so operators can investigate the original file. Both the job update
    and record creation happen in a single transaction.

    Args:
        db: Active SQLAlchemy session.
        job_id: UUID string identifying the failed job.
        reason: Description of the permanent failure (e.g., "MaxRetriesExceeded").
        raw_audio_key: S3 object key for the original uploaded audio.
        error_detail: Full exception string for debugging.

    Raises:
        ValueError: If no job with the given ID exists.
    """
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if not job:
        raise ValueError(f"Job {job_id} not found")

    job.status = "dead_letter"
    job.error_detail = reason

    record = DeadLetterRecord(
        job_id=job_id,
        reason=reason,
        raw_audio_key=raw_audio_key,
        error_detail=error_detail,
    )
    db.add(record)
    db.commit()


def get_quarantine_records(
    db: Session,
    status: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> tuple[list[QuarantineRecord], int]:
    """List quarantine records with optional review status filter and pagination.

    Args:
        db: Active SQLAlchemy session.
        status: Filter by review_status (e.g., "pending_review", "escalated").
        offset: Number of records to skip (for pagination).
        limit: Maximum number of records to return.

    Returns:
        Tuple of (list of QuarantineRecord objects, total count matching filter).
    """
    query = db.query(QuarantineRecord)
    if status:
        query = query.filter(QuarantineRecord.review_status == status)
    total = query.count()
    records = (
        query.order_by(QuarantineRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return records, total


def get_quarantine_record(db: Session, record_id: str) -> QuarantineRecord | None:
    """Retrieve a single quarantine record by its primary key.

    Args:
        db: Active SQLAlchemy session.
        record_id: UUID string of the quarantine record.

    Returns:
        QuarantineRecord if found, None otherwise.
    """
    return db.query(QuarantineRecord).filter(QuarantineRecord.id == record_id).first()


def get_quarantine_record_by_job(db: Session, job_id: str) -> QuarantineRecord | None:
    """Retrieve a quarantine record associated with a specific job.

    Args:
        db: Active SQLAlchemy session.
        job_id: UUID string of the parent job.

    Returns:
        QuarantineRecord if the job was quarantined, None otherwise.
    """
    return db.query(QuarantineRecord).filter(QuarantineRecord.job_id == job_id).first()


def update_quarantine_review(
    db: Session,
    record_id: str,
    reviewer_id: str,
    review_status: str,
    reviewer_notes: str | None = None,
) -> QuarantineRecord | None:
    """Update a quarantine record with a reviewer's decision.

    Also updates the parent job's status based on the review outcome.
    Both updates happen in a single transaction for atomicity.

    Args:
        db: Active SQLAlchemy session.
        record_id: UUID string of the quarantine record to update.
        reviewer_id: Identifier of the human reviewer.
        review_status: One of "escalated", "resolved_false_alarm", "resolved_dispatched".
        reviewer_notes: Optional free-text notes from the reviewer.

    Returns:
        Updated QuarantineRecord if found, None otherwise.
    """
    record = db.query(QuarantineRecord).filter(QuarantineRecord.id == record_id).first()
    if not record:
        return None

    record.reviewer_id = reviewer_id
    record.review_status = review_status
    if reviewer_notes:
        record.reviewer_notes = reviewer_notes

    job = db.query(JobRecord).filter(JobRecord.id == record.job_id).first()
    if job:
        if review_status == "resolved_dispatched":
            job.status = "completed"
        elif review_status == "resolved_false_alarm":
            job.status = "failed"

    db.commit()
    db.refresh(record)
    return record


def get_webhook_config(db: Session, job_id: str) -> dict | None:
    """Retrieve webhook configuration for a job.

    Webhook config is stored in the job's extracted_data JSONB field
    under the "_webhook" key if provided at upload time.

    Args:
        db: Active SQLAlchemy session.
        job_id: UUID string of the job.

    Returns:
        Dict with "url" and optional "secret" keys, or None if no webhook configured.
    """
    job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
    if job and job.extracted_data and "_webhook" in job.extracted_data:
        return job.extracted_data["_webhook"]
    return None


def _increment_org_usage(db: Session, org_id: str, completed: bool = True) -> None:
    """Increment the organization's monthly usage counters.

    Auto-resets the counter if we've crossed into a new billing period.
    Also upserts the monthly UsageRecord for detailed tracking.
    """
    from datetime import datetime, timezone

    org = db.query(Organization).filter(Organization.id == org_id).first()
    if not org:
        return

    now = datetime.now(timezone.utc)

    # Reset if billing period expired (30-day windows)
    if org.billing_period_start is None:
        org.billing_period_start = now
    elif (now - org.billing_period_start).days >= 30:
        org.jobs_used_this_month = 0
        org.billing_period_start = now

    org.jobs_used_this_month += 1

    # Upsert monthly UsageRecord
    usage = (
        db.query(UsageRecord)
        .filter(
            UsageRecord.org_id == org_id,
            UsageRecord.year == now.year,
            UsageRecord.month == now.month,
        )
        .first()
    )
    if not usage:
        usage = UsageRecord(
            org_id=org_id,
            year=now.year,
            month=now.month,
        )
        db.add(usage)

    if completed:
        usage.jobs_completed += 1
    else:
        usage.jobs_failed += 1
