"""Unit tests for SQLAlchemy models."""

from db.models import JobRecord, QuarantineRecord, DeadLetterRecord, utc_now


class TestJobRecord:
    """Tests for the JobRecord model."""

    def test_default_status(self, tmp_path):
        """Should default to pending status."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from db.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        job = JobRecord(id="test-123")
        db.add(job)
        db.commit()

        fetched = db.query(JobRecord).filter(JobRecord.id == "test-123").first()
        assert fetched is not None
        assert fetched.status == "pending"
        assert fetched.extracted_data is None
        assert fetched.error_detail is None

        db.close()

    def test_utc_now_returns_aware_datetime(self):
        """Should return timezone-aware datetime."""
        now = utc_now()
        assert now.tzinfo is not None

    def test_request_id_field(self, tmp_path):
        """Should store request_id."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from db.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        job = JobRecord(id="test-req", request_id="req-abc-123")
        db.add(job)
        db.commit()

        fetched = db.query(JobRecord).filter(JobRecord.id == "test-req").first()
        assert fetched.request_id == "req-abc-123"
        db.close()


class TestQuarantineRecord:
    """Tests for the QuarantineRecord model."""

    def test_default_review_status(self):
        """Should default to pending_review."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from db.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        job = JobRecord(id="job-123")
        db.add(job)
        db.commit()

        record = QuarantineRecord(
            id="qr-123",
            job_id="job-123",
            reason="Low confidence",
            raw_audio_key="incoming/job-123.wav",
        )
        db.add(record)
        db.commit()

        fetched = db.query(QuarantineRecord).filter(QuarantineRecord.id == "qr-123").first()
        assert fetched is not None
        assert fetched.review_status == "pending_review"
        assert fetched.reason == "Low confidence"

        db.close()


class TestDeadLetterRecord:
    """Tests for the DeadLetterRecord model."""

    def test_create_dead_letter(self):
        """Should create a dead letter record."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from db.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        job = JobRecord(id="job-dl-123")
        db.add(job)
        db.commit()

        record = DeadLetterRecord(
            id="dl-123",
            job_id="job-dl-123",
            reason="Max retries exceeded",
            raw_audio_key="incoming/job-dl-123.wav",
            error_detail="Connection timeout",
        )
        db.add(record)
        db.commit()

        fetched = db.query(DeadLetterRecord).filter(DeadLetterRecord.id == "dl-123").first()
        assert fetched is not None
        assert fetched.reason == "Max retries exceeded"
        assert fetched.error_detail == "Connection timeout"

        db.close()
