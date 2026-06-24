import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

Base = declarative_base()

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL", "postgresql://raksha:changeme@db:5432/audio_pipeline")
        kwargs = {"pool_pre_ping": True}
        if url.startswith("postgresql"):
            kwargs.update({
                "pool_size": 20,
                "max_overflow": 10,
                "pool_recycle": 1800,
                "pool_timeout": 30,
            })
        else:
            kwargs["connect_args"] = {"check_same_thread": False}
            kwargs["poolclass"] = StaticPool
        _engine = create_engine(url, **kwargs)
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal


def __getattr__(name):
    if name == "SessionLocal":
        return _get_session_factory()
    if name == "engine":
        return _get_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_db():
    """FastAPI dependency that yields a database session."""
    db = _get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def reset_engine():
    """Reset engine/session for testing with a new DATABASE_URL."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
