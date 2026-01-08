from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

TAG_COLUMNS = ["tagNameList", "shortFeatures", "otherTagList"]
IMAGE_COLUMNS = [
    "coverImageUrl",
    "dynamicCoverImageUrl",
    "imageUrl",
    "imgUrl",
    "image",
    "firstImage",
]
DETAIL_COLUMNS = ["detailUrl", "url", "wapUrl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="首页仪表盘数据聚合脚本")
    parser.add_argument("--fs", default="hdfs://node1:8020")
    parser.add_argument("--metastore_uris", default="thrift://node1:9083")
    parser.add_argument("--database", default="default")
    parser.add_argument("--table", default="dwd_attractions")
    parser.add_argument("--output_db", default="default")
    parser.add_argument("--output", type=Path, default=Path("spark/outputs/index_summary.json"))
    parser.add_argument("--topn", type=int, default=8)
    return parser.parse_args()


def build_spark(app_name: str, fs: str, metastore_uris: Optional[str]) -> SparkSession:
    builder = SparkSession.builder.appName(app_name).enableHiveSupport()
    builder = builder.config("spark.hadoop.fs.defaultFS", fs)
    if metastore_uris:
        builder = builder.config("hive.metastore.uris", metastore_uris)
    return builder.getOrCreate()


def load_table(spark: SparkSession, db: str, table: str) -> DataFrame:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")
    return spark.table(f"{db}.{table}")


def build_cards(df: DataFrame) -> Dict[str, float]:
    total = df.select("poiId").distinct().count()
    paid = df.filter(F.col("price") > 0).select("poiId").distinct().count()
    free = total - paid
    avg_price = df.filter(F.col("price").isNotNull()).agg(F.avg("price")).first()[0] or 0
    avg_heat = df.filter(F.col("heatScore").isNotNull()).agg(F.avg("heatScore")).first()[0] or 0
    return {
        "totalAttractions": int(total),
        "paidAttractions": int(paid),
        "freeAttractions": int(free),
        "avgPrice": round(avg_price, 2),
        "avgHeat": round(avg_heat, 2),
    }


def build_city_overview(df: DataFrame, limit: int) -> List[Dict[str, float]]:
    price_expr = (
        F.when(F.col("price") <= 0, F.lit("免费"))
        .when(F.col("price") < 100, F.lit("0-99"))
        .when(F.col("price") < 200, F.lit("100-199"))
        .when(F.col("price") < 300, F.lit("200-299"))
        .otherwise(F.lit("300+"))
    )
    pdf = (
        df.withColumn("priceBucket", price_expr)
        .groupBy("priceBucket")
        .agg(
            F.count("*").alias("count"),
            F.avg("commentScore").alias("avgScore"),
        )
        .orderBy("priceBucket")
        .toPandas()
    )
    return [
        {
            "bucket": row["priceBucket"],
            "count": int(row["count"] or 0),
            "avgScore": round(row["avgScore"], 2) if row["avgScore"] is not None else None,
        }
        for _, row in pdf.iterrows()
    ]


def build_category_matrix(df: DataFrame) -> List[Dict[str, float]]:
    column = "categoryType" if "categoryType" in df.columns else "tagName"
    if column not in df.columns:
        return []
    pdf = (
        df.groupBy(F.col(column).alias("category"))
        .agg(F.countDistinct("poiId").alias("value"))
        .orderBy(F.desc("value"))
        .limit(10)
        .toPandas()
    )
    return [{"category": row["category"], "value": int(row["value"] or 0)} for _, row in pdf.iterrows()]


def build_price_trend(df: DataFrame) -> List[Dict[str, float]]:
    if "heatScore" not in df.columns:
        return []
    date_col = (
        F.from_unixtime(F.col("timestamp") / 1000).cast("date")
        if "timestamp" in df.columns
        else F.current_date()
    )
    pdf = (
        df.withColumn("event_date", date_col)
        .groupBy("event_date")
        .agg(
            F.avg("heatScore").alias("avgHeat"),
            F.avg("price").alias("avgPrice"),
        )
        .orderBy("event_date")
        .limit(12)
        .toPandas()
    )
    return [
        {
            "date": row["event_date"].isoformat() if row["event_date"] is not None else "",
            "avgHeat": round(row["avgHeat"], 2) if row["avgHeat"] is not None else None,
            "avgPrice": round(row["avgPrice"], 2) if row["avgPrice"] is not None else None,
        }
        for _, row in pdf.iterrows()
    ]


def build_recent_trend(df: DataFrame) -> List[Dict[str, float]]:
    city_col = "crawlDistrictName" if "crawlDistrictName" in df.columns else "districtName"
    if city_col not in df.columns:
        return []
    window = Window.partitionBy(city_col).orderBy(F.col("commentScore").desc_nulls_last())
    ranked = df.withColumn("rank", F.row_number().over(window)).filter(F.col("rank") <= 5)
    grouped = (
        ranked.groupBy(city_col)
        .agg(F.collect_list(F.struct("poiName", "commentScore", "commentCount", "price")).alias("items"))
        .collect()
    )
    result = []
    for row in grouped:
        city = row[city_col]
        for item in row["items"]:
            result.append(
                {
                    "city": city,
                    "poiName": item["poiName"],
                    "commentScore": float(item["commentScore"]) if item["commentScore"] is not None else None,
                    "commentCount": int(item["commentCount"] or 0),
                    "price": float(item["price"]) if item["price"] is not None else None,
                }
            )
    return result


def extract_tags(value: Optional[str]) -> List[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in value.split(",") if item.strip()]


def build_hot_tags(df: DataFrame, limit: int) -> List[Dict[str, float]]:
    tag_col = next((col for col in TAG_COLUMNS if col in df.columns), None)
    if not tag_col:
        return []
    exploded = (
        df.select(tag_col)
        .where(F.col(tag_col).isNotNull())
        .withColumn("tagArray", F.from_json(F.col(tag_col), "array<string>"))
        .select(F.explode(F.coalesce("tagArray", F.split(F.col(tag_col), ","))).alias("tag"))
        .groupBy("tag")
        .agg(F.count("*").alias("value"))
        .orderBy(F.desc("value"))
        .limit(limit)
        .toPandas()
    )
    return [{"name": row["tag"], "value": int(row["value"] or 0)} for _, row in exploded.iterrows()]


def first_non_null(data: Dict[str, object], columns: List[str]) -> Optional[str]:
    for column in columns:
        value = data.get(column)
        if isinstance(value, str) and value.startswith("http"):
            return value
    return None


def _coalesce_columns(df: DataFrame, candidates: List[str]):
    exprs = [F.col(col) for col in candidates if col in df.columns]
    if not exprs:
        exprs = [F.lit(None)]
    return F.coalesce(*exprs)


def build_featured_list(df: DataFrame, limit: int) -> List[Dict[str, object]]:
    image_expr = _coalesce_columns(df, IMAGE_COLUMNS).alias("image")
    link_expr = _coalesce_columns(df, DETAIL_COLUMNS).alias("link")
    desc_expr = _coalesce_columns(df, ["shortDesc", "description", "shortFeatures"]).alias("desc")
    city_col = "crawlDistrictName" if "crawlDistrictName" in df.columns else "districtName"

    select_cols = [
        "poiId",
        "poiName",
        "commentScore",
        "price",
        "heatScore",
        image_expr,
        link_expr,
        F.col(city_col).alias("city") if city_col in df.columns else F.lit(""),
        desc_expr,
    ]
    pdf = (
        df.select(*select_cols)
        .orderBy(F.desc("heatScore"))
        .limit(limit)
        .toPandas()
    )
    return [
        {
            "poiId": int(row["poiId"]),
            "name": row["poiName"],
            "score": row["commentScore"],
            "price": row["price"],
            "heatScore": row["heatScore"],
            "imageUrl": row["image"],
            "detailUrl": row["link"],
            "city": row["city"],
            "desc": row["desc"],
        }
        for _, row in pdf.iterrows()
    ]


SCHEMAS = {
    "indexDash_cards": StructType(
        [
            StructField("totalAttractions", LongType(), True),
            StructField("paidAttractions", LongType(), True),
            StructField("freeAttractions", LongType(), True),
            StructField("avgPrice", DoubleType(), True),
            StructField("avgHeat", DoubleType(), True),
        ]
    ),
    "indexDash_city_overview": StructType(
        [
            StructField("bucket", StringType(), True),
            StructField("count", LongType(), True),
            StructField("avgScore", DoubleType(), True),
        ]
    ),
    "indexDash_category_matrix": StructType(
        [
            StructField("category", StringType(), True),
            StructField("value", LongType(), True),
        ]
    ),
    "indexDash_price_trend": StructType(
        [
            StructField("date", StringType(), True),
            StructField("avgHeat", DoubleType(), True),
            StructField("avgPrice", DoubleType(), True),
        ]
    ),
    "indexDash_recent_trend": StructType(
        [
            StructField("city", StringType(), True),
            StructField("poiName", StringType(), True),
            StructField("commentScore", DoubleType(), True),
            StructField("commentCount", LongType(), True),
            StructField("price", DoubleType(), True),
        ]
    ),
    "indexDash_hot_tags": StructType(
        [
            StructField("name", StringType(), True),
            StructField("value", LongType(), True),
        ]
    ),
    "indexDash_featured_list": StructType(
        [
            StructField("poiId", LongType(), True),
            StructField("name", StringType(), True),
            StructField("score", DoubleType(), True),
            StructField("price", DoubleType(), True),
            StructField("heatScore", DoubleType(), True),
            StructField("imageUrl", StringType(), True),
            StructField("detailUrl", StringType(), True),
            StructField("city", StringType(), True),
            StructField("desc", StringType(), True),
        ]
    ),
}


def write_table(spark: SparkSession, db: str, table: str, records: List[Dict[str, object]]) -> None:
    schema = SCHEMAS[table]
    if records:
        df = spark.createDataFrame(records, schema=schema)
    else:
        df = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)
    df.write.mode("overwrite").saveAsTable(f"{db}.{table}")


def main() -> None:
    args = parse_args()
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spark = build_spark("IndexDashboardAggregator", args.fs, args.metastore_uris)
    try:
        df = load_table(spark, args.database, args.table)
        cards = build_cards(df)
        city = build_city_overview(df, args.topn)
        category = build_category_matrix(df)
        price = build_price_trend(df)
        trend = build_recent_trend(df)
        tags = build_hot_tags(df, 60)
        featured = build_featured_list(df, args.topn)

        payload = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "cards": cards,
            "cityOverview": city,
            "categoryMatrix": category,
            "priceTrend": price,
            "recentTrend": trend,
            "hotTags": tags,
            "featuredList": featured,
        }

        output_db = args.output_db
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {output_db}")
        write_table(spark, output_db, "indexDash_cards", [cards])
        write_table(spark, output_db, "indexDash_city_overview", city)
        write_table(spark, output_db, "indexDash_category_matrix", category)
        write_table(spark, output_db, "indexDash_price_trend", price)
        write_table(spark, output_db, "indexDash_recent_trend", trend)
        write_table(spark, output_db, "indexDash_hot_tags", tags)
        write_table(spark, output_db, "indexDash_featured_list", featured)

        if args.output:
            with output_path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
        print("首页仪表盘数据生成完成")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()


