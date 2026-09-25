"""
SQLite database layer for RideVision Backend.
Manages persistent storage for potholes, user reports, verification confirmations,
and city authority complaint routing configurations.
"""

import sqlite3
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "ridevision.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. City Authority Configurations Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS city_configs (
            city_name TEXT PRIMARY KEY,
            authority_name TEXT NOT NULL,
            channel_type TEXT NOT NULL,
            contact_value TEXT NOT NULL,
            instructions TEXT NOT NULL,
            supports_auto_forward INTEGER DEFAULT 0
        )
    """)

    # 2. Potholes Master Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS potholes (
            id TEXT PRIMARY KEY,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            city TEXT NOT NULL,
            address TEXT,
            severity TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            confirmation_count INTEGER DEFAULT 1,
            fixed_confirmation_count INTEGER DEFAULT 0,
            image_path TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # 3. Individual Commuter Reports Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            pothole_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            severity TEXT NOT NULL,
            image_path TEXT,
            note TEXT,
            reported_at TEXT NOT NULL,
            FOREIGN KEY (pothole_id) REFERENCES potholes(id)
        )
    """)

    # 4. Crowd Verification Confirmations Table (Still There / Fixed)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS confirmations (
            id TEXT PRIMARY KEY,
            pothole_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            confirmation_type TEXT NOT NULL,
            confirmed_at TEXT NOT NULL,
            FOREIGN KEY (pothole_id) REFERENCES potholes(id)
        )
    """)

    conn.commit()
    seed_initial_data(cursor, conn)
    conn.close()


def seed_initial_data(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    # Seed City Configurations
    cities = [
        (
            "Mangaluru",
            "Mangaluru City Corporation (MCC)",
            "whatsapp",
            "919449007722",
            "Send the pothole photo, location link, and description to MCC's official WhatsApp grievance desk.",
            0
        ),
        (
            "Bengaluru",
            "Bruhat Bengaluru Mahanagara Palike (BBMP)",
            "helpline",
            "080-22660000",
            "Call BBMP 24x7 control room or register via BBMP Sahaaya / FixMyStreet app.",
            0
        ),
        (
            "Udupi",
            "Udupi City Municipal Council",
            "helpline",
            "0820-2520306",
            "Contact Udupi CMC Civic Grievance cell directly or report via District portal.",
            0
        ),
        (
            "Mysuru",
            "Mysuru City Corporation (MCC)",
            "helpline",
            "0821-2440890",
            "Dial MCC Mysuru citizen toll-free line or file online grievance.",
            0
        ),
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO city_configs 
        (city_name, authority_name, channel_type, contact_value, instructions, supports_auto_forward)
        VALUES (?, ?, ?, ?, ?, ?)
    """, cities)

    # Seed Sample Potholes for Demonstration & Testing
    cursor.execute("SELECT COUNT(*) FROM potholes")
    if cursor.fetchone()[0] == 0:
        now = datetime.now(timezone.utc).isoformat()
        sample_potholes = [
            # Mangaluru coordinates (Near SJEC Vamanjoor, Kadri, Hampankatta, Kankanady, Pumpwell)
            (
                "pothole-mng-001",
                12.9152,
                74.8988,
                "Mangaluru",
                "NH 73 near SJEC Gate, Vamanjoor",
                "severe",
                "active",
                6,
                0,
                "/static/uploads/pothole_sjec.jpg",
                now,
                now
            ),
            (
                "pothole-mng-002",
                12.8715,
                74.8564,
                "Mangaluru",
                "Kankanady Bypass Road near Father Muller",
                "severe",
                "active",
                8,
                0,
                "/static/uploads/pothole_kankanady.jpg",
                now,
                now
            ),
            (
                "pothole-mng-003",
                12.8798,
                74.8532,
                "Mangaluru",
                "Kadri Temple Road, Mallikatte Junction",
                "moderate",
                "active",
                4,
                1,
                "/static/uploads/pothole_kadri.jpg",
                now,
                now
            ),
            (
                "pothole-mng-004",
                12.8682,
                74.8427,
                "Mangaluru",
                "Hampankatta Circle near City Bus Stand",
                "minor",
                "reported_fixed",
                2,
                3,
                None,
                now,
                now
            ),
            (
                "pothole-mng-005",
                12.8615,
                74.8650,
                "Mangaluru",
                "Pumpwell Flyover Service Road",
                "severe",
                "active",
                9,
                0,
                None,
                now,
                now
            ),
            # Bengaluru coordinates (Indiranagar, Koramangala, Silk Board)
            (
                "pothole-blr-001",
                12.9719,
                77.6412,
                "Bengaluru",
                "100 Feet Road, HAL 2nd Stage, Indiranagar",
                "severe",
                "active",
                12,
                0,
                None,
                now,
                now
            ),
            (
                "pothole-blr-002",
                12.9352,
                77.6245,
                "Bengaluru",
                "80 Feet Road, 4th Block, Koramangala",
                "moderate",
                "active",
                5,
                1,
                None,
                now,
                now
            ),
            (
                "pothole-blr-003",
                12.9176,
                77.6238,
                "Bengaluru",
                "Hosur Road, Central Silk Board Junction",
                "severe",
                "verified_fixed",
                3,
                7,
                None,
                now,
                now
            ),
        ]

        cursor.executemany("""
            INSERT INTO potholes
            (id, lat, lon, city, address, severity, status, confirmation_count, fixed_confirmation_count, image_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_potholes)

    conn.commit()


# --- Database Query Functions ---

def get_all_potholes(status: Optional[str] = None, city: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    query = "SELECT * FROM potholes WHERE 1=1"
    params = []

    if status:
        query += " AND status = ?"
        params.append(status)
    if city:
        query += " AND LOWER(city) = LOWER(?)"
        params.append(city)

    query += " ORDER BY updated_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_pothole_by_id(pothole_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM potholes WHERE id = ?", (pothole_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def save_or_merge_pothole(
    lat: float,
    lon: float,
    city: str,
    severity: str,
    user_id: str,
    image_path: Optional[str] = None,
    note: Optional[str] = None,
    address: Optional[str] = None,
    merge_threshold_m: float = 15.0
) -> Dict[str, Any]:
    """
    Saves a new pothole or merges with an existing pothole if within merge_threshold_m (15 meters).
    This handles deduplication cleanly at the persistence level.
    """
    from geo_matching import haversine_distance_m, bounding_box

    conn = get_db_connection()
    cursor = conn.cursor()

    # Pre-filter candidate potholes using bounding box
    min_lat, max_lat, min_lon, max_lon = bounding_box(lat, lon, merge_threshold_m)
    cursor.execute("""
        SELECT * FROM potholes
        WHERE status != 'verified_fixed'
          AND lat BETWEEN ? AND ?
          AND lon BETWEEN ? AND ?
    """, (min_lat, max_lat, min_lon, max_lon))
    candidates = cursor.fetchall()

    target_pothole_id = None
    is_new = False
    now = datetime.now(timezone.utc).isoformat()

    # Check exact Haversine distance
    for row in candidates:
        dist = haversine_distance_m(lat, lon, row["lat"], row["lon"])
        if dist <= merge_threshold_m:
            target_pothole_id = row["id"]
            break

    if target_pothole_id:
        # Existing pothole found: increment confirmation count and bump severity if higher
        cursor.execute("""
            UPDATE potholes
            SET confirmation_count = confirmation_count + 1,
                updated_at = ?,
                status = CASE WHEN status = 'reported_fixed' THEN 'active' ELSE status END
            WHERE id = ?
        """, (now, target_pothole_id))
    else:
        # Create brand new pothole
        target_pothole_id = f"pothole-{uuid.uuid4().hex[:8]}"
        is_new = True
        cursor.execute("""
            INSERT INTO potholes
            (id, lat, lon, city, address, severity, status, confirmation_count, fixed_confirmation_count, image_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'active', 1, 0, ?, ?, ?)
        """, (target_pothole_id, lat, lon, city, address or f"Road near {lat:.4f}, {lon:.4f}", severity, image_path, now, now))

    # Record individual report
    report_id = f"rep-{uuid.uuid4().hex[:8]}"
    cursor.execute("""
        INSERT INTO reports
        (id, pothole_id, user_id, lat, lon, severity, image_path, note, reported_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (report_id, target_pothole_id, user_id, lat, lon, severity, image_path, note, now))

    conn.commit()

    # Fetch updated record
    updated_pothole = cursor.execute("SELECT * FROM potholes WHERE id = ?", (target_pothole_id,)).fetchone()
    conn.close()

    return {
        "pothole": dict(updated_pothole),
        "report_id": report_id,
        "is_new": is_new
    }


def record_confirmation(user_id: str, pothole_id: str, confirmation_type: str) -> Dict[str, Any]:
    """
    Records a verification: 'still_there' or 'fixed'.
    Updates counts and triggers status transitions.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    conf_id = f"conf-{uuid.uuid4().hex[:8]}"
    cursor.execute("""
        INSERT INTO confirmations (id, pothole_id, user_id, confirmation_type, confirmed_at)
        VALUES (?, ?, ?, ?, ?)
    """, (conf_id, pothole_id, user_id, confirmation_type, now))

    if confirmation_type == "still_there":
        cursor.execute("""
            UPDATE potholes
            SET confirmation_count = confirmation_count + 1,
                status = 'active',
                updated_at = ?
            WHERE id = ?
        """, (now, pothole_id))
    elif confirmation_type == "fixed":
        cursor.execute("""
            UPDATE potholes
            SET fixed_confirmation_count = fixed_confirmation_count + 1,
                status = CASE 
                    WHEN fixed_confirmation_count + 1 >= 3 THEN 'verified_fixed'
                    ELSE 'reported_fixed'
                END,
                updated_at = ?
            WHERE id = ?
        """, (now, pothole_id))

    conn.commit()
    updated = cursor.execute("SELECT * FROM potholes WHERE id = ?", (pothole_id,)).fetchone()
    conn.close()

    return {
        "confirmation_id": conf_id,
        "pothole": dict(updated) if updated else None
    }


def get_last_user_confirmation(user_id: str, pothole_id: str) -> Optional[datetime]:
    conn = get_db_connection()
    row = conn.execute("""
        SELECT confirmed_at FROM confirmations
        WHERE user_id = ? AND pothole_id = ?
        ORDER BY confirmed_at DESC LIMIT 1
    """, (user_id, pothole_id)).fetchone()
    conn.close()

    if row and row["confirmed_at"]:
        try:
            return datetime.fromisoformat(row["confirmed_at"].replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def get_city_config(city_name: Optional[str]) -> Dict[str, Any]:
    default = {
        "city_name": city_name or "Unknown",
        "authority_name": "Local Road Maintenance Authority",
        "channel_type": "none",
        "contact_value": "",
        "instructions": "No municipal forwarding channel configured yet. Report is saved in RideVision map.",
        "supports_auto_forward": 0
    }
    if not city_name:
        return default

    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM city_configs WHERE LOWER(city_name) = LOWER(?)",
        (city_name.strip(),)
    ).fetchone()
    conn.close()

    return dict(row) if row else default


def get_analytics_summary() -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()

    total_potholes = cursor.execute("SELECT COUNT(*) FROM potholes").fetchone()[0]
    active_potholes = cursor.execute("SELECT COUNT(*) FROM potholes WHERE status = 'active'").fetchone()[0]
    reported_fixed = cursor.execute("SELECT COUNT(*) FROM potholes WHERE status = 'reported_fixed'").fetchone()[0]
    verified_fixed = cursor.execute("SELECT COUNT(*) FROM potholes WHERE status = 'verified_fixed'").fetchone()[0]
    total_reports = cursor.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    total_confirmations = cursor.execute("SELECT COUNT(*) FROM confirmations").fetchone()[0]

    severity_counts = dict(cursor.execute(
        "SELECT severity, COUNT(*) FROM potholes GROUP BY severity"
    ).fetchall())

    city_counts = dict(cursor.execute(
        "SELECT city, COUNT(*) FROM potholes GROUP BY city"
    ).fetchall())

    conn.close()

    return {
        "total_potholes": total_potholes,
        "active_potholes": active_potholes,
        "reported_fixed": reported_fixed,
        "verified_fixed": verified_fixed,
        "total_reports": total_reports,
        "total_confirmations": total_confirmations,
        "severity_distribution": severity_counts,
        "city_distribution": city_counts,
        "fix_rate_percent": round((verified_fixed / total_potholes * 100), 1) if total_potholes > 0 else 0.0
    }


def update_pothole_status(pothole_id: str, new_status: str) -> bool:
    """Update status of a pothole (active, reported_fixed, verified_fixed)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        UPDATE potholes
        SET status = ?, updated_at = ?
        WHERE id = ?
    """, (new_status, now, pothole_id))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def get_all_reports(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent individual reports."""
    conn = get_db_connection()
    rows = conn.execute("""
        SELECT * FROM reports ORDER BY reported_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_or_update_city_config(
    city_name: str,
    authority_name: str,
    channel_type: str,
    contact_value: str,
    instructions: str,
    supports_auto_forward: int = 0
):
    """Upsert city routing configuration."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO city_configs (city_name, authority_name, channel_type, contact_value, instructions, supports_auto_forward)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(city_name) DO UPDATE SET
            authority_name = excluded.authority_name,
            channel_type = excluded.channel_type,
            contact_value = excluded.contact_value,
            instructions = excluded.instructions,
            supports_auto_forward = excluded.supports_auto_forward
    """, (city_name, authority_name, channel_type, contact_value, instructions, supports_auto_forward))
    conn.commit()
    conn.close()


def reset_database():
    """Drop and reseed the database to initial clean state."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS confirmations")
    cursor.execute("DROP TABLE IF EXISTS reports")
    cursor.execute("DROP TABLE IF EXISTS potholes")
    cursor.execute("DROP TABLE IF EXISTS city_configs")
    conn.commit()
    conn.close()
    init_db()

