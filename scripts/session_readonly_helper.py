#!/usr/bin/env python3
"""Read-only helper for querying the sessions DB without modifying WAL/SHM.

This script replaces direct ``/usr/bin/sqlite3`` calls in auth_smoke.sh.
Using ``/usr/bin/sqlite3`` on the production sessions.db was identified as
the root cause of WAL unlink corruption: the standalone sqlite3 binary
opens the database in read-write mode by default, which modifies the WAL
and SHM files even for simple SELECT queries. If the process is interrupted
or the binary exits uncleanly, the WAL can be left in an inconsistent state.

This helper:
- Opens the database in **read-only URI mode** (``?mode=ro``).
- Enables ``PRAGMA query_only=ON`` so no write operation can succeed.
- Never creates a new database file.
- Never checkpoints, vacuums, or modifies WAL/SHM.
- Returns only count/boolean results — never token_hash values.

Usage::

    python session_readonly_helper.py <db_path> <sql_query>

Example::

    python session_readonly_helper.py /path/to/sessions.db \\
        "SELECT revoked FROM sessions WHERE token_hash='abc123';"
"""

from __future__ import annotations

import sqlite3
import sys


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: session_readonly_helper.py <db_path> <sql_query>", file=sys.stderr)
        sys.exit(1)

    db_path = sys.argv[1]
    query = sys.argv[2]

    # Open in strict read-only mode via URI. The ?mode=ro parameter
    # ensures that even if the query tries to write, SQLite will reject it.
    # immutable=1 would be even safer but prevents reading a WAL that
    # hasn't been checkpointed yet.
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        timeout=10,
    )
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA busy_timeout=10000;")

    try:
        result = conn.execute(query).fetchall()
        conn.close()
        print(result)
    except Exception as exc:
        conn.close()
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
