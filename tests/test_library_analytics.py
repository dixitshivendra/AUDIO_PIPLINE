"""Integration tests for library search and analytics endpoints."""

import hashlib
import json
import uuid

import pytest
from sqlalchemy import text

from db.database import _get_session_factory
from db.models import ApiKey, Organization


@pytest.fixture
def test_org_id():
    """Create a test organization and return its ID."""
    org_id = str(uuid.uuid4())
    SessionLocal = _get_session_factory()
    db = SessionLocal()
    org = Organization(id=org_id, name="Test Org", slug=f"test-{uuid.uuid4().hex[:8]}")
    db.add(org)
    db.commit()
    db.close()
    return org_id


@pytest.fixture
def tenant_api_key(test_org_id):
    """Create a per-tenant API key and return the raw key string."""
    raw_key = f"test-tenant-{uuid.uuid4().hex}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    SessionLocal = _get_session_factory()
    db = SessionLocal()
    api_key = ApiKey(
        org_id=test_org_id,
        key_hash=key_hash,
        key_prefix=raw_key[:8],
        name="test-key",
        is_active=True,
    )
    db.add(api_key)
    db.commit()
    db.close()
    return raw_key


class TestLibrarySearch:
    """Tests for the searchable library endpoint."""

    def test_library_returns_empty_when_no_completed_jobs(self, client, api_key):
        """Library should return empty list when no completed jobs exist."""
        resp = client.get("/api/v1/library", headers={"X-API-Key": api_key})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["jobs"] == []

    def test_library_search_by_keyword(self, client, tenant_api_key, db_session, test_org_id):
        """Full-text search should find jobs matching keywords."""
        job_id = str(uuid.uuid4())
        db_session.execute(text('''
            INSERT INTO jobs (id, org_id, status, title, transcript, summary, sentiment,
                            sentiment_score, keywords, topics, action_items, decisions,
                            language, speakers, tags, created_at, updated_at)
            VALUES (:id, :org_id, 'completed', :title, :transcript, :summary, :sentiment,
                    0.8, :keywords, :topics, :action_items, :decisions,
                    :language, :speakers, :tags, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        '''), {
            'id': job_id,
            'org_id': test_org_id,
            'title': 'Q2 Planning Meeting',
            'transcript': 'We discussed the API integration timeline and security audit findings.',
            'summary': 'Planning meeting about API and security.',
            'sentiment': 'positive',
            'keywords': json.dumps(['API', 'security', 'integration']),
            'topics': json.dumps(['planning', 'security']),
            'action_items': json.dumps(['Complete API by Friday']),
            'decisions': json.dumps(['Accelerate timeline']),
            'language': 'en',
            'speakers': json.dumps(['Alice', 'Bob']),
            'tags': json.dumps(['planning', 'Q2']),
        })
        db_session.commit()

        resp = client.get("/api/v1/library?q=security", headers={"X-API-Key": tenant_api_key})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["id"] == job_id
        assert "security" in data["jobs"][0]["summary"].lower()

    def test_library_filter_by_sentiment(self, client, tenant_api_key, db_session, test_org_id):
        """Sentiment filter should return only matching jobs."""
        pos_id = str(uuid.uuid4())
        db_session.execute(text('''
            INSERT INTO jobs (id, org_id, status, sentiment, sentiment_score,
                            created_at, updated_at)
            VALUES (:id, :org_id, 'completed', 'positive', 0.9,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        '''), {'id': pos_id, 'org_id': test_org_id})

        neg_id = str(uuid.uuid4())
        db_session.execute(text('''
            INSERT INTO jobs (id, org_id, status, sentiment, sentiment_score,
                            created_at, updated_at)
            VALUES (:id, :org_id, 'completed', 'negative', 0.2,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        '''), {'id': neg_id, 'org_id': test_org_id})
        db_session.commit()

        resp = client.get("/api/v1/library?sentiment=positive", headers={"X-API-Key": tenant_api_key})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["id"] == pos_id

    def test_library_detail_returns_full_data(self, client, tenant_api_key, db_session, test_org_id):
        """Library detail should return all fields including transcript."""
        job_id = str(uuid.uuid4())
        db_session.execute(text('''
            INSERT INTO jobs (id, org_id, status, title, transcript, summary,
                            sentiment, sentiment_score, keywords, topics, action_items,
                            decisions, language, speakers, tags, created_at, updated_at)
            VALUES (:id, :org_id, 'completed', 'Test Call', 'Hello world transcript.',
                    'A test call summary.', 'neutral', 0.5,
                    :keywords, :topics, :action_items, :decisions,
                    'en', :speakers, :tags, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        '''), {
            'id': job_id, 'org_id': test_org_id,
            'keywords': json.dumps(['test']),
            'topics': json.dumps(['testing']),
            'action_items': json.dumps(['Do something']),
            'decisions': json.dumps(['Decided something']),
            'speakers': json.dumps(['Speaker 1']),
            'tags': json.dumps(['test']),
        })
        db_session.commit()

        resp = client.get(f"/api/v1/library/{job_id}", headers={"X-API-Key": tenant_api_key})
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == job_id
        assert data["title"] == "Test Call"
        assert data["transcript"] == "Hello world transcript."
        assert data["summary"] == "A test call summary."
        assert "test" in data["keywords"]


class TestAnalytics:
    """Tests for analytics endpoints."""

    def test_analytics_overview_empty(self, client, api_key):
        """Analytics overview should work with no data."""
        resp = client.get("/api/v1/analytics/overview", headers={"X-API-Key": api_key})
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert "sentiment" in data
        assert data["jobs"]["total"] == 0

    def test_analytics_overview_with_data(self, client, tenant_api_key, db_session, test_org_id):
        """Analytics overview should aggregate job data correctly."""
        for i in range(3):
            job_id = str(uuid.uuid4())
            sentiment = "positive" if i < 2 else "negative"
            score = 0.8 if i < 2 else 0.2
            db_session.execute(text('''
                INSERT INTO jobs (id, org_id, status, sentiment, sentiment_score,
                                created_at, updated_at)
                VALUES (:id, :org_id, 'completed', :sentiment, :score,
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            '''), {'id': job_id, 'org_id': test_org_id, 'sentiment': sentiment, 'score': score})
        db_session.commit()

        resp = client.get("/api/v1/analytics/overview?days=30", headers={"X-API-Key": tenant_api_key})
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"]["total"] == 3
        assert data["jobs"]["completed"] == 3
        assert data["sentiment"]["positive"] == 2
        assert data["sentiment"]["negative"] == 1

    def test_top_keywords_empty(self, client, api_key):
        """Top keywords should return empty list with no data."""
        resp = client.get("/api/v1/analytics/top-keywords", headers={"X-API-Key": api_key})
        assert resp.status_code == 200
        data = resp.json()
        assert "keywords" in data

    def test_top_keywords_with_data(self, client, tenant_api_key, db_session, test_org_id):
        """Top keywords should rank by frequency."""
        job_id = str(uuid.uuid4())
        db_session.execute(text('''
            INSERT INTO jobs (id, org_id, status, keywords, created_at, updated_at)
            VALUES (:id, :org_id, 'completed', :keywords, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        '''), {
            'id': job_id, 'org_id': test_org_id,
            'keywords': json.dumps(['api', 'security', 'api', 'deploy', 'api']),
        })
        db_session.commit()

        resp = client.get("/api/v1/analytics/top-keywords?days=30", headers={"X-API-Key": tenant_api_key})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["keywords"]) > 0
        assert data["keywords"][0]["keyword"] == "api"
        assert data["keywords"][0]["count"] == 3
