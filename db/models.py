"""SQLAlchemy ORM models for the RAKSHA audio pipeline.

Defines the persistence schema for users, organizations, jobs, quarantine
records, billing, and dead letter queue. All models use UTC timestamps
and UUID primary keys.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON,
)
from sqlalchemy.orm import relationship

from db.database import Base


def utc_now() -> datetime:
    """Return the current UTC time with timezone info."""
    return datetime.now(timezone.utc)


# ── Auth & Multi-tenancy ──────────────────────────────────────────────


class User(Base):
    """User account with email/password authentication."""

    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_superadmin = Column(Boolean, default=False, nullable=False)
    avatar_url = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    memberships = relationship("Membership", back_populates="user", cascade="all, delete-orphan")


class Organization(Base):
    """Tenant organization that owns jobs and billing."""

    __tablename__ = "organizations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    logo_url = Column(String(512), nullable=True)
    plan = Column(String(20), nullable=False, default='free')
    stripe_customer_id = Column(String(255), nullable=True, unique=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    monthly_job_limit = Column(Integer, default=100, nullable=False)
    jobs_used_this_month = Column(Integer, default=0, nullable=False)
    billing_period_start = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    memberships = relationship("Membership", back_populates="organization", cascade="all, delete-orphan")
    jobs = relationship("JobRecord", back_populates="organization")
    usage_records = relationship("UsageRecord", back_populates="organization", cascade="all, delete-orphan")


class Membership(Base):
    """Links users to organizations with a role."""

    __tablename__ = "memberships"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    org_id = Column(String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False, default='member')
    created_at = Column(DateTime(timezone=True), default=utc_now)

    user = relationship("User", back_populates="memberships")
    organization = relationship("Organization", back_populates="memberships")


# ── Audio Pipeline ─────────────────────────────────────────────────────


class JobRecord(Base):
    """Tracks the lifecycle of an audio processing job.

    Status flow: pending -> processing -> completed | quarantine | failed | dead_letter
    """

    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(
        String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=True, index=True,
    )
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    status = Column(String(20), nullable=False, default='pending', index=True)
    extracted_data = Column(JSON, nullable=True)
    error_detail = Column(Text, nullable=True)
    request_id = Column(String, nullable=True)
    # Library/search columns
    filename = Column(String(512), nullable=True)
    title = Column(String(512), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    file_size = Column(Integer, nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    transcript = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    speakers = Column(JSON, nullable=True)
    sentiment = Column(String(20), nullable=True, index=True)
    sentiment_score = Column(Float, nullable=True)
    keywords = Column(JSON, nullable=True)
    topics = Column(JSON, nullable=True)
    action_items = Column(JSON, nullable=True)
    decisions = Column(JSON, nullable=True)
    language = Column(String(10), nullable=True, index=True)
    tags = Column(JSON, nullable=True)
    extra_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, index=True)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    organization = relationship("Organization", back_populates="jobs")
    quarantine_record = relationship(
        "QuarantineRecord", back_populates="job", uselist=False, cascade="all, delete-orphan",
    )
    dead_letter_record = relationship(
        "DeadLetterRecord", back_populates="job", uselist=False, cascade="all, delete-orphan",
    )


class QuarantineRecord(Base):
    """Records audio jobs that require human review."""

    __tablename__ = "quarantine_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(
        String, ForeignKey("jobs.id", ondelete="CASCADE"), unique=True, nullable=False,
    )
    reason = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=True)
    transcript_excerpt = Column(String, nullable=True)
    raw_audio_key = Column(String, nullable=False)
    clean_audio_key = Column(String, nullable=True)
    review_status = Column(String(30), nullable=False, default='pending_review')
    reviewer_id = Column(String, nullable=True)
    reviewer_notes = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    job = relationship("JobRecord", back_populates="quarantine_record")

    def to_response(self) -> dict:
        """Serialize to API response dict."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "reason": self.reason,
            "confidence_score": self.confidence_score,
            "transcript_excerpt": self.transcript_excerpt,
            "raw_audio_key": self.raw_audio_key,
            "clean_audio_key": self.clean_audio_key,
            "review_status": self.review_status,
            "reviewer_id": self.reviewer_id,
            "reviewer_notes": self.reviewer_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DeadLetterRecord(Base):
    """Records jobs that permanently failed after exhausting all retries."""

    __tablename__ = "dead_letter_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(
        String, ForeignKey("jobs.id", ondelete="CASCADE"), unique=True, nullable=False,
    )
    reason = Column(String, nullable=False)
    raw_audio_key = Column(String, nullable=True)
    error_detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    job = relationship("JobRecord", back_populates="dead_letter_record")


# ── Billing & Usage ────────────────────────────────────────────────────


class UsageRecord(Base):
    """Tracks per-org monthly API usage for billing and enforcement."""

    __tablename__ = "usage_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(
        String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False,
    )
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    jobs_completed = Column(Integer, default=0, nullable=False)
    jobs_failed = Column(Integer, default=0, nullable=False)
    audio_seconds = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    organization = relationship("Organization", back_populates="usage_records")


class ApiKey(Base):
    """Per-tenant API key for programmatic access."""

    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(
        String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False,
    )
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    key_prefix = Column(String(8), nullable=False)
    name = Column(String(255), nullable=False, default="default")
    is_active = Column(Boolean, default=True, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    organization = relationship("Organization")


class AuditLog(Base):
    """Audit trail for org-level actions."""

    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(
        String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)


class PasswordReset(Base):
    """Password reset token tracking."""

    __tablename__ = "password_resets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)


# ── Library, Alerts, Webhooks ──────────────────────────────────────────


class AlertRule(Base):
    """Alert rule configuration for an organization."""

    __tablename__ = "alert_rules"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(
        String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    name = Column(String(255), nullable=False)
    rule_type = Column(String(50), nullable=False)
    config = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    alerts = relationship("Alert", back_populates="rule", cascade="all, delete-orphan")


class Alert(Base):
    """Triggered alert instance."""

    __tablename__ = "alerts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(
        String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    rule_id = Column(String, ForeignKey("alert_rules.id", ondelete="SET NULL"), nullable=True)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=True)
    severity = Column(String(20), nullable=False, index=True)
    title = Column(String(512), nullable=False)
    message = Column(Text, nullable=True)
    matched_text = Column(Text, nullable=True)
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, index=True)

    rule = relationship("AlertRule", back_populates="alerts")


class Webhook(Base):
    """Webhook endpoint configuration for an organization."""

    __tablename__ = "webhooks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id = Column(
        String, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    name = Column(String(255), nullable=False)
    url = Column(String(1024), nullable=False)
    secret = Column(String(512), nullable=True)
    events = Column(JSON, nullable=False, default=list)
    is_active = Column(Boolean, default=True, nullable=False)
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    failure_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)


class Comment(Base):
    """Comment on a job by a user."""

    __tablename__ = "comments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(
        String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
