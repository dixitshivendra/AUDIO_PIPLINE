"""Tests for auth routes: register, login, refresh, forgot-password, verify-email."""

import uuid
import pytest


class TestRegister:
    def test_register_success(self, client):
        email = f"reg-{uuid.uuid4().hex[:8]}@example.com"
        response = client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "strongpassword1",
            "full_name": "Test User",
        })
        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["email"] == email
        assert data["user"]["is_superadmin"] is False

    def test_register_duplicate_email(self, client):
        email = f"dup-{uuid.uuid4().hex[:8]}@example.com"
        client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "strongpassword1",
        })
        response = client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "strongpassword2",
        })
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    def test_register_weak_password_too_short(self, client):
        email = f"weak-{uuid.uuid4().hex[:8]}@example.com"
        response = client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "short",
        })
        assert response.status_code == 422
        assert "8 characters" in response.json()["detail"]

    def test_register_all_numeric_password(self, client):
        email = f"num-{uuid.uuid4().hex[:8]}@example.com"
        response = client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "12345678",
        })
        assert response.status_code == 422
        assert "numeric" in response.json()["detail"]

    def test_register_invalid_email(self, client):
        response = client.post("/api/v1/auth/register", json={
            "email": "not-an-email",
            "password": "strongpassword1",
        })
        assert response.status_code == 422


class TestLogin:
    def test_login_success(self, client):
        email = f"login-{uuid.uuid4().hex[:8]}@example.com"
        password = "testpassword123"
        client.post("/api/v1/auth/register", json={
            "email": email,
            "password": password,
        })
        response = client.post("/api/v1/auth/login", json={
            "email": email,
            "password": password,
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["user"]["email"] == email

    def test_login_wrong_password(self, client):
        email = f"wrong-{uuid.uuid4().hex[:8]}@example.com"
        client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "correctpassword1",
        })
        response = client.post("/api/v1/auth/login", json={
            "email": email,
            "password": "wrongpassword1",
        })
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        response = client.post("/api/v1/auth/login", json={
            "email": f"nonexistent-{uuid.uuid4().hex[:8]}@example.com",
            "password": "anypassword1",
        })
        assert response.status_code == 401


class TestRefreshToken:
    def test_refresh_success(self, client):
        email = f"refresh-{uuid.uuid4().hex[:8]}@example.com"
        reg = client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "testpassword123",
        })
        refresh_token = reg.json()["refresh_token"]
        response = client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token,
        })
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_refresh_with_access_token_fails(self, client):
        email = f"refresh2-{uuid.uuid4().hex[:8]}@example.com"
        reg = client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "testpassword123",
        })
        access_token = reg.json()["access_token"]
        response = client.post("/api/v1/auth/refresh", json={
            "refresh_token": access_token,
        })
        assert response.status_code == 401


class TestGetMe:
    def test_me_authenticated(self, client, auth_headers):
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        assert "email" in response.json()

    def test_me_unauthenticated(self, client):
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401


class TestForgotPassword:
    def test_forgot_password_returns_message(self, client):
        email = f"forgot-{uuid.uuid4().hex[:8]}@example.com"
        client.post("/api/v1/auth/register", json={
            "email": email,
            "password": "testpassword123",
        })
        response = client.post("/api/v1/auth/forgot-password", json={
            "email": email,
        })
        assert response.status_code == 200
        assert "reset link" in response.json()["message"].lower()
        # Token should NOT be in the response
        assert "reset_url" not in response.json()

    def test_forgot_password_nonexistent_email(self, client):
        response = client.post("/api/v1/auth/forgot-password", json={
            "email": f"nonexistent-{uuid.uuid4().hex[:8]}@example.com",
        })
        assert response.status_code == 200
        assert "reset link" in response.json()["message"].lower()


class TestVerifyEmail:
    def test_verify_email_invalid_token(self, client):
        response = client.post("/api/v1/auth/verify-email", json={
            "token": "invalid-token",
        })
        assert response.status_code == 400
