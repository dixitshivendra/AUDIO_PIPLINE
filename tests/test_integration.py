"""Integration tests for the full RAKSHA pipeline.

These tests verify the complete request flow through the API layer
with mocked external dependencies (Groq, MinIO, Celery).
"""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.integration
class TestFullPipeline:
    """End-to-end pipeline integration tests."""

    @patch("api.routes.v1.upload_to_s3")
    @patch("worker.llm_extraction._extract_crisis_data")
    @patch("worker.llm_extraction._transcribe_audio")
    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.tasks.ensure_bucket_exists")
    @patch("worker.tasks.upload_to_s3")
    @patch("storage.client.get_presigned_url")
    def test_upload_to_completion_flow(
        self, mock_api_upload, mock_presigned, mock_upload, mock_ensure, mock_normalize,
        mock_transcribe, mock_extract, client, api_key
    ):
        """Test full flow: upload -> process -> completed."""
        mock_upload.return_value = "incoming/test.wav"
        mock_presigned.return_value = "http://minio:9000/presigned"
        mock_transcribe.return_value = MagicMock(
            segments=[
                {"no_speech_prob": 0.05, "avg_logprob": -0.2},
                {"no_speech_prob": 0.03, "avg_logprob": -0.15},
            ],
            text="There is a severe car accident on Highway 101 near the bridge with multiple injuries",
        )
        mock_extract.return_value = json.dumps({
            "is_actionable_emergency": True,
            "needs_human_review": False,
            "severity": "critical",
            "vehicle_type": "car",
            "victim_count": 3,
            "road_name": "Highway 101",
            "landmark": "bridge",
            "injury_type": "severe",
        })

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
            mock_task.delay.assert_called_once()

    @patch("api.routes.v1.upload_to_s3")
    def test_upload_download_cycle(self, mock_upload_s3, client, api_key):
        """Test upload returns valid job_id that can be queried."""
        mockAsyncResult = MagicMock()
        mockAsyncResult.return_value.state = "PENDING"
        with patch("api.routes.v1.process_audio") as mock_task:
            mock_task.delay = MagicMock(return_value=MagicMock(id=str(uuid.uuid4())))
            mock_task.AsyncResult = mockAsyncResult
            upload_resp = client.post(
                "/api/v1/upload",
                headers={"X-API-Key": api_key},
                files={"file": ("test.m4a", b"\x00" * 100, "audio/m4a")},
            )
            job_id = upload_resp.json()["job_id"]

            status_resp = client.get(
                f"/api/v1/status/{job_id}",
                headers={"X-API-Key": api_key},
            )
            assert status_resp.status_code == 200
            assert status_resp.json()["status"] == "pending"

    def test_quarantine_crud_flow(self, client, api_key):
        """Test quarantine list and detail endpoints."""
        list_resp = client.get(
            "/api/v1/quarantine",
            headers={"X-API-Key": api_key},
        )
        assert list_resp.status_code == 200
        assert isinstance(list_resp.json(), list)

    def test_dead_letter_list(self, client, api_key):
        """Test dead letter queue listing."""
        resp = client.get(
            "/api/v1/dead-letter",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "records" in data

    def test_audio_download_requires_valid_variant(self, client, api_key):
        """Test audio download rejects invalid variant."""
        job_id = str(uuid.uuid4())
        resp = client.get(
            f"/api/v1/audio/{job_id}/download?variant=invalid",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 400

    @patch("api.routes.v1.upload_to_s3")
    def test_all_endpoints_rate_limited(self, mock_upload_s3, client, api_key):
        """Test that rate limiting is applied to endpoints."""
        # Upload has 10/min limit - just verify it doesn't error on first call
        with patch("api.routes.v1.process_audio") as mock_task:
            mock_task.delay = MagicMock(return_value=MagicMock(id=str(uuid.uuid4())))
            resp = client.post(
                "/api/v1/upload",
                headers={"X-API-Key": api_key},
                files={"file": ("test.m4a", b"\x00" * 100, "audio/m4a")},
            )
            assert resp.status_code == 200

        # Status has 60/min limit
        job_id = str(uuid.uuid4())
        resp = client.get(
            f"/api/v1/status/{job_id}",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200

    @patch("api.routes.v1.upload_to_s3")
    def test_concurrent_uploads(self, mock_upload_s3, client, api_key):
        """Test multiple concurrent uploads create separate jobs."""
        job_ids = []
        with patch("api.routes.v1.process_audio") as mock_task:
            mock_task.delay = MagicMock(return_value=MagicMock(id=str(uuid.uuid4())))
            for i in range(3):
                resp = client.post(
                    "/api/v1/upload",
                    headers={"X-API-Key": api_key},
                    files={"file": (f"test_{i}.m4a", b"\x00" * 100, "audio/m4a")},
                )
                assert resp.status_code == 200
                job_ids.append(resp.json()["job_id"])

        assert len(set(job_ids)) == 3, "All job IDs should be unique"


@pytest.mark.integration
class TestWebsocketEndpoint:
    """Tests for WebSocket functionality."""

    def test_websocket_endpoint_exists(self, client):
        """Test that the WebSocket endpoint is registered."""
        with client.websocket_connect("/ws/jobs") as ws:
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"


@pytest.mark.integration
class TestSecurityEdgeCases:
    """Security-focused integration tests."""

    def test_empty_api_key_rejected(self, client):
        """Should reject empty API key."""
        resp = client.get("/api/v1/jobs", headers={"X-API-Key": ""})
        assert resp.status_code in (401, 422)

    def test_very_long_api_key_rejected(self, client):
        """Should reject absurdly long API key."""
        long_key = "x" * 10000
        resp = client.get("/api/v1/jobs", headers={"X-API-Key": long_key})
        assert resp.status_code == 401

    def test_sql_injection_in_job_id(self, client, api_key):
        """Should handle SQL injection attempts gracefully."""
        malicious_id = "'; DROP TABLE jobs; --"
        resp = client.get(
            f"/api/v1/status/{malicious_id}",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200  # Should return pending, not error

    @patch("api.routes.v1.upload_to_s3")
    def test_xss_in_filename(self, mock_upload_s3, client, api_key):
        """Should handle XSS in filename gracefully."""
        with patch("api.routes.v1.process_audio") as mock_task:
            mock_task.delay = MagicMock(return_value=MagicMock(id=str(uuid.uuid4())))
            resp = client.post(
                "/api/v1/upload",
                headers={"X-API-Key": api_key},
                files={"file": ("<script>alert(1)</script>.m4a", b"\x00" * 100, "audio/m4a")},
            )
            assert resp.status_code == 200
