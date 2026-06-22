import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB, ENUM
from sqlalchemy.orm import relationship
from db.database import Base


def utc_now():
    return datetime.now(timezone.utc)


# Strict Enum Types
JobStatus = ENUM('pending', 'processing', 'completed', 'quarantine',
                 'failed', name='job_status_enum', create_type=False)
ReviewStatus = ENUM('pending_review', 'escalated', 'resolved_false_alarm',
                    'resolved_dispatched', name='review_status_enum', create_type=False)


class JobRecord(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(JobStatus, nullable=False, default='pending')
    extracted_data = Column(JSONB, nullable=True)
    error_detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True),
                        default=utc_now, onupdate=utc_now)

    quarantine_record = relationship(
        "QuarantineRecord", back_populates="job", uselist=False, cascade="all, delete-orphan")


class QuarantineRecord(Base):
    __tablename__ = "quarantine_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey(
        "jobs.id", ondelete="CASCADE"), unique=True, nullable=False)
    reason = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=True)
    transcript_excerpt = Column(String, nullable=True)
    raw_audio_key = Column(String, nullable=False)
    clean_audio_key = Column(String, nullable=True)
    review_status = Column(ReviewStatus, nullable=False,
                           default='pending_review')
    reviewer_id = Column(String, nullable=True)
    reviewer_notes = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True),
                        default=utc_now, onupdate=utc_now)

    job = relationship("JobRecord", back_populates="quarantine_record")
