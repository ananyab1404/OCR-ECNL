from collections.abc import Generator

from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


settings = get_settings()

engine = create_engine(settings["database_url"], pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@event.listens_for(engine, "connect")
def _register_pgvector(dbapi_connection, connection_record) -> None:
    register_vector(dbapi_connection)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()