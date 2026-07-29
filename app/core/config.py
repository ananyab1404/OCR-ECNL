import os
from functools import lru_cache
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_settings() -> dict[str, str | int | None]:
    database_url = os.getenv("Database_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Database_URL or DATABASE_URL is required")

    db_host = os.getenv("DB_HOST")
    if db_host:
        parsed = urlparse(database_url)
        if parsed.scheme and parsed.netloc and "@" in parsed.netloc:
            userinfo, _, _ = parsed.netloc.rpartition("@")
            database_url = parsed._replace(netloc=f"{userinfo}@{db_host}").geturl()

    return {
        "database_url": database_url,
        "openai_api_key": os.getenv("OPEN_AI") or os.getenv("OPENAI_API_KEY"),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_dimensions": int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")),
    }
