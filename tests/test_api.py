"""API integration tests for RAKSHA endpoints."""

import uuid
from unittest.mock import MagicMock, patch

import pytest


class TestHealthEndpoints:
    """Tests for health and readiness probes."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness_check(self, client):
        response = client.get("/ready")
        # In test env, DB/Redis/MinIO may not be available, so 503 is acceptable
        assert response.status_code in (200, 503)
        assert response.json()["status"] in ("ready", "degraded")

    def test_dashboard_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "RAKSHA" in response.text


class TestUploadEndpoint:
    """Tests for audio upload endpoint."""

    def test_upload_requires_api_key(self, client):
        response = client.post("/api/v1/upload", files={"file": ("test.m4a", b"data", "audio/m4a")})
        assert response.status_code in (401, 422)

    def test_upload_rejects_invalid_key(self, client):
        response = client.post(
            "/api/v1/upload",
            headers={"X-API-Key": "wrong-key"},
            files={"file": ("test.m4a", b"data", "audio/m4a")},
        )
        assert response.status_code == 401

    def test_upload_rejects_invalid_file_type(self, client, api_key):
        response = client.post(
            "/api/v1/upload",
            headers={"X-API-Key": api_key},
            files={"file": ("test.exe", b"data", "application/octet-stream")},
        )
        assert response.status_code == 400

    @patch("api.routes.v1.upload_to_s3")
    def test_upload_accepts_valid_audio(self, mock_upload_s3, client, api_key):
        with patch("api.routes.v1.process_audio") as mock_task:
            mock_task.delay = MagicMock(return_value=MagicMock(id=str(uuid.uuid4())))
            response = client.post(
                "/api/v1/upload",
                headers={"X-API-Key": api_key},
                files={"file": ("emergency.m4a", b"\x00" * 100, "audio/m4a")},
            )
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "pending"

    def test_upload_rejects_oversized_file(self, client, api_key):
        large_content = b"\x00" * (50 * 1024 * 1024 + 1)
        response = client.post(
            "/api/v1/upload",
            headers={"X-API-Key": api_key},
            files={"file": ("huge.m4a", large_content, "audio/m4a")},
        )
        assert response.status_code == 413

    @patch("api.routes.v1.upload_to_s3")
    def test_upload_with_webhook_params(self, mock_upload_s3, client, api_key):
        with patch("api.routes.v1.process_audio") as mock_task:
            mock_task.delay = MagicMock(return_value=MagicMock(id=str(uuid.uuid4())))
            response = client.post(
                "/api/v1/upload?webhook_url=https://example.com/hook&webhook_secret=sec123",
                headers={"X-API-Key": api_key},
                files={"file": ("emergency.m4a", b"\x00" * 100, "audio/m4a")},
            )
            assert response.status_code == 200


class TestStatusEndpoint:
    """Tests for job status endpoint."""

    def test_status_requires_api_key(self, client):
        job_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/status/{job_id}")
        assert response.status_code in (401, 422)

    def test_status_returns_pending_for_unknown_job(self, client, api_key):
        job_id = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/status/{job_id}",
            headers={"X-API-Key": api_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"


class TestJobsEndpoint:
    """Tests for job listing endpoint."""

    def test_jobs_requires_api_key(self, client):
        response = client.get("/api/v1/jobs")
        assert response.status_code in (401, 422)

    def test_jobs_returns_list(self, client, api_key):
        response = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": api_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "jobs" in data
        assert isinstance(data["jobs"], list)

    def test_jobs_filter_by_status(self, client, api_key):
        response = client.get(
            "/api/v1/jobs?status=completed&limit=5",
            headers={"X-API-Key": api_key},
        )
        assert response.status_code == 200


class TestQuarantineEndpoints:
    """Tests for quarantine management endpoints."""

    def test_quarantine_list_requires_api_key(self, client):
        response = client.get("/api/v1/quarantine")
        assert response.status_code in (401, 422)

    def test_quarantine_list_returns_list(self, client, api_key):
        response = client.get(
            "/api/v1/quarantine",
            headers={"X-API-Key": api_key},
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_quarantine_detail_not_found(self, client, api_key):
        fake_id = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/quarantine/{fake_id}",
            headers={"X-API-Key": api_key},
        )
        assert response.status_code == 404

    def test_review_requires_valid_status(self, client, api_key):
        fake_id = str(uuid.uuid4())
        response = client.patch(
            f"/api/v1/quarantine/{fake_id}/review",
            headers={"X-API-Key": api_key},
            json={
                "reviewer_id": "reviewer-1",
                "review_status": "invalid_status",
            },
        )
        assert response.status_code == 400

    def test_review_not_found(self, client, api_key):
        fake_id = str(uuid.uuid4())
        response = client.patch(
            f"/api/v1/quarantine/{fake_id}/review",
            headers={"X-API-Key": api_key},
            json={
                "reviewer_id": "reviewer-1",
                "review_status": "escalated",
                "reviewer_notes": "Needs specialist review",
            },
        )
        assert response.status_code == 404


class TestAudioDownloadEndpoint:
    """Tests for audio download endpoint."""

    def test_download_requires_api_key(self, client):
        job_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/audio/{job_id}/download")
        assert response.status_code in (401, 422)

    def test_download_rejects_invalid_variant(self, client, api_key):
        job_id = str(uuid.uuid4())
        response = client.get(
            f"/api/v1/audio/{job_id}/download?variant=invalid",
            headers={"X-API-Key": api_key},
        )
        assert response.status_code == 400


class TestDeadLetterEndpoint:
    """Tests for dead letter queue endpoint."""

    def test_dead_letter_requires_api_key(self, client):
        response = client.get("/api/v1/dead-letter")
        assert response.status_code in (401, 422)

    def test_dead_letter_returns_list(self, client, api_key):
        response = client.get(
            "/api/v1/dead-letter",
            headers={"X-API-Key": api_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "records" in data


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_returns_prometheus_format(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
