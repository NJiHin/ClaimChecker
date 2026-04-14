import asyncpg


async def init_db(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS passages (
            id           TEXT    PRIMARY KEY,
            doc_id       TEXT    NOT NULL,
            doc_title    TEXT    NOT NULL,
            text         TEXT    NOT NULL,
            context_text TEXT,
            embedding    vector(3072),
            page_start   INTEGER,
            created_at   BIGINT  NOT NULL
        )
    """)


    await conn.execute("""
        CREATE TABLE IF NOT EXISTS claim_cache (
            sentence_hash  TEXT    NOT NULL,
            kb_state_hash  TEXT    NOT NULL,
            sentence_text  TEXT    NOT NULL,
            result_json    TEXT    NOT NULL,
            checked_at     BIGINT  NOT NULL,
            PRIMARY KEY (sentence_hash, kb_state_hash)
        )
    """)


async def prune_claim_cache(conn: asyncpg.Connection) -> None:
    seven_days_ms = 7 * 24 * 60 * 60 * 1000
    import time
    cutoff = int(time.time() * 1000) - seven_days_ms
    await conn.execute("DELETE FROM claim_cache WHERE checked_at < $1", cutoff)
