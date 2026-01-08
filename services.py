from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Tuple

from django.conf import settings
from pyhive import hive


@contextmanager
def hive_cursor():
    """Provide a Hive cursor using settings.HIVE_CONFIG."""
    cfg = settings.HIVE_CONFIG
    connection = hive.Connection(
        host=cfg.get("host"),
        port=cfg.get("port", 10000),
        username=cfg.get("username"),
        password=(cfg.get("password") or None),
        database=cfg.get("database", "default"),
    )
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        connection.close()


def rows_to_dicts(columns: Iterable[Tuple[str]], rows: Iterable[Tuple[Any]]) -> List[Dict[str, Any]]:
    titles = [col[0] for col in columns]
    result = []
    for row in rows:
        record = {}
        for idx, value in enumerate(row):
            if isinstance(value, Decimal):
                record[titles[idx]] = float(value)
            else:
                record[titles[idx]] = value
        result.append(record)
    return result


def build_where_clause(params: Dict[str, Any]) -> Tuple[str, List[Any]]:
    clauses: List[str] = []
    values: List[Any] = []
    if keyword := (params.get("keyword") or "").strip():
        like = f"%{keyword.lower()}%"
        clauses.append(
            "("
            "lower(poiName) LIKE %s OR "
            "lower(zoneName) LIKE %s OR "
            "lower(districtName) LIKE %s OR "
            "lower(shortFeatures) LIKE %s OR "
            "lower(tagNameList) LIKE %s"
            ")"
        )
        values.extend([like] * 5)
    if district := (params.get("district") or "").strip():
        if str(district).isdigit():
            clauses.append("districtId = %s")
            values.append(int(district))
        else:
            clauses.append("lower(districtName) LIKE %s")
            values.append(f"%{district.lower()}%")
    if sight_level := params.get("sightLevel"):
        if sight_level == "其他":
            clauses.append(
                "(lower(sightLevelStr) NOT IN ('5a', '4a', '3a') OR sightLevelStr IS NULL)"
            )
        else:
            clauses.append("lower(sightLevelStr) = %s")
            values.append(sight_level.lower())
    if params.get("minPrice") not in (None, "", "null"):
        clauses.append("price >= %s")
        values.append(float(params["minPrice"]))
    if params.get("maxPrice") not in (None, "", "null"):
        clauses.append("price <= %s")
        values.append(float(params["maxPrice"]))
    if tag := (params.get("tagName") or "").strip():
        clauses.append("lower(tagNameList) LIKE %s")
        values.append(f"%{tag.lower()}%")
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_sql, values


def fetch_attractions(
    page: int,
    page_size: int,
    filters: Dict[str, Any],
) -> Dict[str, Any]:
    table = settings.HIVE_CONFIG.get("table", "dwd_attractions")
    offset = max(page - 1, 0) * page_size

    where_sql, values = build_where_clause(filters)

    count_sql = f"SELECT COUNT(1) FROM {table} {where_sql}"
    query_sql = f"""
        SELECT
            poiId,
            poiName,
            districtId,
            districtName,
            zoneName,
            sightLevelStr,
            commentScore,
            commentCount,
            heatScore,
            shortFeatures,
            tagNameList,
            priceTagList,
            price,
            marketPrice,
            marketDiscountPrice,
            underlineDiscountPrice,
            priceTypeDesc,
            detailUrl,
            coverImageUrl,
            timestamp,
            discount_rate
        FROM {table}
        {where_sql}
        ORDER BY heatScore DESC
        LIMIT {page_size} OFFSET {offset}
    """
    with hive_cursor() as cursor:
        cursor.execute(count_sql, values)
        total = int(cursor.fetchone()[0])
        cursor.execute(query_sql, values)
        rows = cursor.fetchall()
        data = rows_to_dicts(cursor.description, rows)
    return {
        "count": total,
        "results": data,
        "page": page,
        "page_size": page_size,
    }


def fetch_attraction_detail(poi_id: int) -> Dict[str, Any] | None:
    table = settings.HIVE_CONFIG.get("table", "dwd_attractions")
    sql = f"SELECT * FROM {table} WHERE poiId = %s LIMIT 1"
    with hive_cursor() as cursor:
        cursor.execute(sql, (poi_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return rows_to_dicts(cursor.description, [row])[0]

