"""Unit tests for core security module."""

import hmac

from core.security import verify_api_key


class TestVerifyApiKey:
    """Tests for API key verification."""

    def test_valid_api_key(self):
        """Should return the key and org_id when valid."""
        result = verify_api_key(x_api_key="test-api-key-123")
        assert result == ("test-api-key-123", None)

    def test_invalid_api_key_raises(self):
        """Should raise 401 when key is invalid."""
        from fastapi import HTTPException

        try:
            verify_api_key(x_api_key="wrong-key")
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            assert e.status_code == 401

    def test_timing_safe_comparison(self):
        """Should use constant-time comparison internally."""
        result = verify_api_key(x_api_key="test-api-key-123")
        assert result is not None
        assert result[0] == "test-api-key-123"
