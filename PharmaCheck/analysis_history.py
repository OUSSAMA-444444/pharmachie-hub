"""Historique local et persistant des analyses de lots, avec export CSV."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import pandas as pd

from pharmacheck_utils import FEATURES, ID, MODEL_NAMES, ROOT

HISTORY_PATH = ROOT / "results/historique_lots.sqlite3"
HISTORY_COLUMNS = [
    "id_analyse", "analyse_le_utc", ID, "resultat_final", "modele_recommande",
    *FEATURES,
    *[column for slug in MODEL_NAMES for column in (f"prediction_{slug}", f"score_{slug}")],
    "desaccord_modeles", "signature_modeles",
]


def record_analysis(frame, predictions, recommended, signature, path=None):
    """Append once per submitted analysis; repeated lot IDs remain separate analyses."""
    path = Path(path or HISTORY_PATH)
    record = {
        "id_analyse": uuid4().hex,
        "analyse_le_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        ID: str(frame.iloc[0][ID]),
        "resultat_final": "À examiner" if predictions.iloc[0][f"prediction_{recommended}"] else "Conforme",
        "modele_recommande": MODEL_NAMES[recommended],
        **{feature: float(frame.iloc[0][feature]) for feature in FEATURES},
        **{f"prediction_{slug}": int(predictions.iloc[0][f"prediction_{slug}"]) for slug in MODEL_NAMES},
        **{f"score_{slug}": float(predictions.iloc[0][f"score_{slug}"]) for slug in MODEL_NAMES},
        "desaccord_modeles": bool(predictions.iloc[0]["desaccord_modeles"]),
        "signature_modeles": signature,
    }
    payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=10)) as connection, connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS analyses "
            "(id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO analyses VALUES (?, ?, ?)",
                           (record["id_analyse"], record["analyse_le_utc"], payload))
    return record["id_analyse"]


def read_history(path=None):
    path = Path(path or HISTORY_PATH)
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    with closing(sqlite3.connect(path, timeout=10)) as connection:
        rows = connection.execute("SELECT payload FROM analyses ORDER BY created_utc DESC, id DESC").fetchall()
    return pd.DataFrame([json.loads(row[0]) for row in rows], columns=HISTORY_COLUMNS)


def history_csv(frame):
    """UTF-8 for Excel; user-entered lot identifiers remain plain text."""
    exported = frame.copy()
    if ID in exported:
        exported[ID] = exported[ID].map(
            lambda value: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else value
        )
    return exported.to_csv(index=False).encode("utf-8-sig")


def delete_history(analysis_ids=None, path=None):
    """Move exact analysis IDs (or all analyses) to a recoverable deletion batch."""
    path = Path(path or HISTORY_PATH)
    if not path.exists() or analysis_ids == []:
        return 0
    with closing(sqlite3.connect(path, timeout=10)) as connection, connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS analyses_supprimees "
            "(id TEXT PRIMARY KEY, created_utc TEXT NOT NULL, payload TEXT NOT NULL, "
            "suppression_id TEXT NOT NULL, deleted_utc TEXT NOT NULL)"
        )
        where = "" if analysis_ids is None else " WHERE id IN (" + ",".join("?" for _ in analysis_ids) + ")"
        params = [] if analysis_ids is None else list(analysis_ids)
        connection.execute(
            "INSERT INTO analyses_supprimees SELECT id, created_utc, payload, ?, ? FROM analyses" + where,
            [uuid4().hex, datetime.now(timezone.utc).isoformat(timespec="microseconds"), *params],
        )
        cursor = connection.execute("DELETE FROM analyses" + where, params)
        return cursor.rowcount


def latest_history_deletion(path=None):
    """Return the latest recoverable batch, including after a browser reload."""
    path = Path(path or HISTORY_PATH)
    if not path.exists():
        return None
    with closing(sqlite3.connect(path, timeout=10)) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='analyses_supprimees'").fetchone():
            return None
        return connection.execute(
            "SELECT suppression_id, COUNT(*) FROM analyses_supprimees GROUP BY suppression_id "
            "ORDER BY MAX(deleted_utc) DESC LIMIT 1"
        ).fetchone()


def restore_history(deletion_id, path=None):
    """Restore exactly one deletion batch with original IDs, timestamps and measurements."""
    path = Path(path or HISTORY_PATH)
    if latest_history_deletion(path) is None:
        return 0
    with closing(sqlite3.connect(path, timeout=10)) as connection, connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO analyses SELECT id, created_utc, payload FROM analyses_supprimees "
            "WHERE suppression_id = ?", (deletion_id,),
        )
        connection.execute("DELETE FROM analyses_supprimees WHERE suppression_id = ?", (deletion_id,))
        return cursor.rowcount
