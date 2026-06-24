"""Shared test fixtures for RAKSHA pipeline tests."""

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """Set required environment variables for all tests."""
    monkeypatch.setenv("AASIOM_DISPATCH_API_KEY", "test-api-key-123")
    monkeypatch.setenv("OPENAI_API_KEY", "gsk_test_key")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-key-for-testing-only-12345678")
    monkeypatch.setenv("FRONTEND_URL", "http://localhost:3000")
    monkeypatch.setenv("DISABLE_RATE_LIMITING", "1")
    monkeypatch.setenv("SMTP_HOST", "localhost")
    monkeypatch.setenv("SMTP_PORT", "25")
    monkeypatch.setenv("SMTP_USER", "")
    monkeypatch.setenv("SMTP_PASSWORD", "")


@pytest.fixture(autouse=True)
def _reset_db_engine():
    """Reset the DB engine before each test so it picks up the env vars."""
    from db.database import reset_engine
    reset_engine()
    yield
    reset_engine()


@pytest.fixture
def db_session():
    """Create a database session for test data insertion."""
    from db.database import _get_session_factory
    SessionLocal = _get_session_factory()
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def api_key():
    """Return the test API key."""
    return "test-api-key-123"


@pytest.fixture
def client():
    """Create a FastAPI test client with tables created."""
    from db.database import _get_engine, Base
    from api.main import app

    engine = _get_engine()
    Base.metadata.create_all(bind=engine)

    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Register a test user and return auth headers."""
    email = f"test-{uuid.uuid4().hex[:8]}@example.com"
    password = "testpassword123"
    response = client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Test User",
    })
    assert response.status_code == 201
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(client):
    """Create a superadmin user and return auth headers."""
    from db.database import _get_session_factory
    from db.models import User
    from core.auth import hash_password

    email = f"admin-{uuid.uuid4().hex[:8]}@example.com"
    password = "adminpassword123"

    SessionLocal = _get_session_factory()
    db = SessionLocal()
    user = User(
        email=email,
        hashed_password=hash_password(password),
        full_name="Test Admin",
        is_active=True,
        is_verified=True,
        is_superadmin=True,
    )
    db.add(user)
    db.commit()
    db.close()

    response = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": password,
    })
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a minimal valid audio file for testing."""
    audio_path = tmp_path / "test_audio.m4a"
    audio_path.write_bytes(b"\x00" * 1024)
    return str(audio_path)


@pytest.fixture
def mock_groq_client():
    """Mock the OpenAI/Groq client for LLM tests."""
    with patch("worker.llm_extraction.get_client") as mock:
        yield mock


@pytest.fixture
def mock_ffmpeg():
    """Mock FFmpeg subprocess calls."""
    with patch("worker.audio_ops.subprocess.run") as mock:
        mock.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
        yield mock


@pytest.fixture
def mock_celery():
    """Mock Celery task dispatch."""
    with patch("worker.tasks.process_audio") as mock:
        mock.delay = MagicMock(return_value=MagicMock(id=str(uuid.uuid4())))
        yield mock


@pytest.fixture
def mock_s3():
    """Mock S3/MinIO client."""
    with patch("storage.client.get_s3_client") as mock:
        s3_mock = MagicMock()
        mock.return_value = s3_mock
        yield s3_mock


@pytest.fixture
def mock_upload_to_s3():
    """Mock S3 upload function."""
    with patch("api.routes.v1.upload_to_s3") as mock:
        mock.return_value = "incoming/test.wav"
        yield mock
