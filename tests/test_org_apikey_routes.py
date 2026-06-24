"""Tests for org management and API key routes."""

import uuid
import pytest


class TestOrgManagement:
    def test_create_org(self, client, auth_headers):
        response = client.post("/api/v1/orgs", headers=auth_headers, json={
            "name": "Test Org",
        })
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["name"] == "Test Org"

    def test_list_orgs(self, client, auth_headers):
        client.post("/api/v1/orgs", headers=auth_headers, json={"name": "Org1"})
        response = client.get("/api/v1/orgs", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_org_detail(self, client, auth_headers):
        create_resp = client.post("/api/v1/orgs", headers=auth_headers, json={"name": "Detail Org"})
        org_id = create_resp.json()["id"]
        response = client.get(f"/api/v1/orgs/{org_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["name"] == "Detail Org"

    def test_update_org(self, client, auth_headers):
        create_resp = client.post("/api/v1/orgs", headers=auth_headers, json={"name": "Update Org"})
        org_id = create_resp.json()["id"]
        response = client.patch(f"/api/v1/orgs/{org_id}", headers=auth_headers, json={
            "name": "Updated Org",
        })
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Org"


class TestOrgMembers:
    def test_list_members(self, client, auth_headers):
        create_resp = client.post("/api/v1/orgs", headers=auth_headers, json={"name": "Member Org"})
        org_id = create_resp.json()["id"]
        response = client.get(f"/api/v1/orgs/{org_id}/members", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_invite_member(self, client, auth_headers):
        create_resp = client.post("/api/v1/orgs", headers=auth_headers, json={"name": "Invite Org"})
        org_id = create_resp.json()["id"]
        invite_email = f"invite-{uuid.uuid4().hex[:8]}@example.com"
        # Register the invited user first
        client.post("/api/v1/auth/register", json={
            "email": invite_email,
            "password": "inviteduser123",
        })
        response = client.post(f"/api/v1/orgs/{org_id}/members", headers=auth_headers, json={
            "email": invite_email,
            "role": "member",
        })
        assert response.status_code == 201


class TestApiKeys:
    def test_create_api_key(self, client, auth_headers):
        response = client.post("/api/v1/api-keys", headers=auth_headers, json={
            "name": "test-key",
        })
        assert response.status_code == 201
        data = response.json()
        assert "key" in data
        assert data["name"] == "test-key"

    def test_list_api_keys(self, client, auth_headers):
        client.post("/api/v1/api-keys", headers=auth_headers, json={"name": "list-key"})
        response = client.get("/api/v1/api-keys", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_revoke_api_key(self, client, auth_headers):
        create_resp = client.post("/api/v1/api-keys", headers=auth_headers, json={
            "name": "revoke-key",
        })
        key_id = create_resp.json()["id"]
        response = client.delete(f"/api/v1/api-keys/{key_id}", headers=auth_headers)
        assert response.status_code == 200
