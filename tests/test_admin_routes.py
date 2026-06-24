"""Tests for admin routes: stats, users, orgs, jobs, audit logs."""

import uuid
import pytest


class TestAdminStats:
    def test_stats_requires_superadmin(self, client, auth_headers):
        response = client.get("/api/admin/stats", headers=auth_headers)
        assert response.status_code == 403

    def test_stats_returns_data(self, client, admin_headers):
        response = client.get("/api/admin/stats", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_orgs" in data
        assert "total_users" in data
        assert "total_jobs" in data
        assert "jobs_by_status" in data
        assert "orgs_by_plan" in data


class TestAdminUsers:
    def test_list_users(self, client, admin_headers):
        response = client.get("/api/admin/users", headers=admin_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_users_requires_superadmin(self, client, auth_headers):
        response = client.get("/api/admin/users", headers=auth_headers)
        assert response.status_code == 403

    def test_create_and_get_user(self, client, admin_headers):
        email = f"admin-create-{uuid.uuid4().hex[:8]}@example.com"
        create_resp = client.post("/api/admin/users", headers=admin_headers, json={
            "email": email,
            "password": "createduser123",
            "full_name": "Created User",
        })
        assert create_resp.status_code == 200
        user_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/admin/users/{user_id}", headers=admin_headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["email"] == email

    def test_toggle_user_active(self, client, admin_headers):
        email = f"toggle-{uuid.uuid4().hex[:8]}@example.com"
        create_resp = client.post("/api/admin/users", headers=admin_headers, json={
            "email": email,
            "password": "toggleuser123",
        })
        user_id = create_resp.json()["id"]

        # Disable
        patch_resp = client.patch(f"/api/admin/users/{user_id}", headers=admin_headers, json={
            "is_active": False,
        })
        assert patch_resp.status_code == 200

        # Verify disabled
        get_resp = client.get(f"/api/admin/users/{user_id}", headers=admin_headers)
        assert get_resp.json()["is_active"] is False


class TestAdminOrgs:
    def test_list_orgs(self, client, admin_headers):
        response = client.get("/api/admin/orgs", headers=admin_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_and_get_org(self, client, admin_headers):
        slug = f"test-org-{uuid.uuid4().hex[:8]}"
        create_resp = client.post("/api/admin/orgs", headers=admin_headers, json={
            "name": "Test Org",
            "slug": slug,
            "plan": "free",
        })
        assert create_resp.status_code == 200
        org_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/admin/orgs/{org_id}", headers=admin_headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["slug"] == slug

    def test_update_org_plan(self, client, admin_headers):
        slug = f"plan-org-{uuid.uuid4().hex[:8]}"
        create_resp = client.post("/api/admin/orgs", headers=admin_headers, json={
            "name": "Plan Org",
            "slug": slug,
        })
        org_id = create_resp.json()["id"]

        patch_resp = client.patch(f"/api/admin/orgs/{org_id}", headers=admin_headers, json={
            "plan": "pro",
            "monthly_job_limit": 10000,
        })
        assert patch_resp.status_code == 200

    def test_delete_org(self, client, admin_headers):
        slug = f"del-org-{uuid.uuid4().hex[:8]}"
        create_resp = client.post("/api/admin/orgs", headers=admin_headers, json={
            "name": "Delete Org",
            "slug": slug,
        })
        org_id = create_resp.json()["id"]

        del_resp = client.delete(f"/api/admin/orgs/{org_id}", headers=admin_headers)
        assert del_resp.status_code == 200


class TestAdminJobs:
    def test_list_jobs(self, client, admin_headers):
        response = client.get("/api/admin/jobs", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "jobs" in data


class TestAdminAuditLogs:
    def test_list_audit_logs(self, client, admin_headers):
        response = client.get("/api/admin/audit-logs", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "logs" in data
