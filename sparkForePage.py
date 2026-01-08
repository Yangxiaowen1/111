from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from numbers import Integral, Real
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="前台多页面可视化数据聚合脚本")
    parser.add_argument("--fs", default="hdfs://node1:8020")
    parser.add_argument("--metastore_uris", default="thrift://node1:9083")
    parser.add_argument("--database", default="default")
    parser.add_argument("--table", default="dwd_attractions")
    parser.add_argument("--output_db", default="default")
    parser.add_argument("--output", type=Path, default=Path("spark/outputs/fore_page_summary.json"))
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


def prepare_dataframe(df: DataFrame) -> DataFrame:
    array_schema = ArrayType(StringType())
    tag_schema = ArrayType(
        StructType(
            [
                StructField("name", StringType(), True),
                StructField("type", StringType(), True),
                StructField("code", StringType(), True),
            ]
        )
    )

    distance_value = F.regexp_extract(F.col("distanceStr"), r"([0-9.]+)", 1).cast(DoubleType())
    distance_in_km = F.when(F.lower(F.col("distanceStr")).contains("m"), distance_value / 1000).otherwise(distance_value)

    return (
        df.withColumn("price", F.col("price").cast(DoubleType()))
        .withColumn("marketPrice", F.col("marketPrice").cast(DoubleType()))
        .withColumn("commentScore", F.col("commentScore").cast(DoubleType()))
        .withColumn("heatScore", F.col("heatScore").cast(DoubleType()))
        .withColumn("timestamp", F.col("timestamp").cast(LongType()))
        .withColumn("tags_array", F.from_json(F.col("tagNameList"), array_schema))
        .withColumn("short_array", F.from_json(F.col("shortFeatures"), array_schema))
        .withColumn("price_tags", F.from_json(F.col("priceTagList"), tag_schema))
        .withColumn("other_tags", F.from_json(F.col("otherTagList"), tag_schema))
        .withColumn("booking_tags", F.from_json(F.col("otherTagList"), tag_schema))
        .withColumn("distanceKm", distance_in_km)
    )


def to_records(pdf) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for _, row in pdf.iterrows():
        data = {}
        for col in pdf.columns:
            value = row[col]
            if isinstance(value, bool):
                data[col] = bool(value)
            elif isinstance(value, Integral):
                data[col] = int(value)
            elif isinstance(value, Real):
                data[col] = round(float(value), 4)
            else:
                data[col] = value
        records.append(data)
    return records


def build_filter_page(df: DataFrame, limit: int = 50) -> Dict[str, List[Dict[str, object]]]:
    city_col = "crawlDistrictName" if "crawlDistrictName" in df.columns else "districtName"
    city_pdf = (
        df.groupBy(city_col)
        .agg(
            F.count("*").alias("attractionCount"),
            F.avg("price").alias("avgPrice"),
            F.avg("heatScore").alias("avgHeat"),
            F.avg(F.when(F.col("price") > 0, 1).otherwise(0)).alias("paidRate"),
        )
        .orderBy(F.desc("attractionCount"))
        .limit(limit)
        .toPandas()
    )
    city_pdf.rename(columns={city_col: "city"}, inplace=True)

    level_pdf = (
        df.where(F.col("sightLevelStr").isNotNull())
        .groupBy("sightLevelStr")
        .agg(
            F.count("*").alias("count"),
            F.avg("commentScore").alias("avgScore"),
            F.avg("price").alias("avgPrice"),
        )
        .orderBy(F.desc("count"))
        .toPandas()
    )

    tags_pdf = (
        df.select(F.explode(F.coalesce("tags_array", F.array(F.lit("")))).alias("tag"), "heatScore")
        .where(F.col("tag").isNotNull() & (F.col("tag") != ""))
        .groupBy("tag")
        .agg(F.count("*").alias("value"), F.avg("heatScore").alias("avgHeat"))
        .orderBy(F.desc("value"))
        .limit(80)
        .toPandas()
    )

    booking_default = F.array(
        F.struct(
            F.lit("未知").alias("name"),
            F.lit("").alias("type"),
            F.lit("").alias("code"),
        )
    )
    booking_pdf = (
        df.select(F.explode(F.coalesce("booking_tags", booking_default)).alias("b"))
        .select("b.name")
        .where(F.col("name").isNotNull())
        .groupBy("name")
        .count()
        .orderBy(F.desc("count"))
        .toPandas()
    )

    price_pdf = (
        df.withColumn(
            "priceBucket",
            F.when(F.col("price") <= 0, F.lit("免费"))
            .when(F.col("price") < 100, F.lit("0-99"))
            .when(F.col("price") < 200, F.lit("100-199"))
            .when(F.col("price") < 300, F.lit("200-299"))
            .when(F.col("price") < 500, F.lit("300-499"))
            .otherwise(F.lit("500+")),
        )
        .groupBy("priceBucket")
        .agg(F.count("*").alias("count"), F.avg("commentScore").alias("avgScore"))
        .orderBy("priceBucket")
        .toPandas()
    )

    return {
        "forePage_filter_city": to_records(city_pdf),
        "forePage_filter_level": to_records(level_pdf),
        "forePage_filter_tags": to_records(tags_pdf),
        "forePage_filter_booking": to_records(booking_pdf),
        "forePage_filter_price": to_records(price_pdf),
    }


def build_price_page(df: DataFrame) -> Dict[str, List[Dict[str, object]]]:
    price_pdf = (
        df.withColumn(
            "bucket",
            F.when(F.col("price") <= 0, F.lit("免费"))
            .when(F.col("price") < 100, F.lit("0-99"))
            .when(F.col("price") < 200, F.lit("100-199"))
            .when(F.col("price") < 300, F.lit("200-299"))
            .when(F.col("price") < 500, F.lit("300-499"))
            .otherwise(F.lit("500+")),
        )
        .groupBy("bucket")
        .agg(F.count("*").alias("count"))
        .orderBy("bucket")
        .toPandas()
    )

    discount_pdf = (
        df.withColumn("discount", F.when(F.col("marketPrice") > 0, F.col("marketPrice") - F.col("price")).otherwise(0.0))
        .withColumn(
            "bucket",
            F.when(F.col("discount") <= 0, F.lit("无优惠"))
            .when(F.col("discount") < 50, F.lit("0-49"))
            .when(F.col("discount") < 100, F.lit("50-99"))
            .when(F.col("discount") < 200, F.lit("100-199"))
            .otherwise(F.lit("200+")),
        )
        .groupBy("bucket")
        .agg(F.count("*").alias("count"), F.avg("discount").alias("avgDiscount"))
        .orderBy("bucket")
        .toPandas()
    )

    scatter_pdf = (
        df.select("price", "commentScore")
        .where(F.col("price").isNotNull() & F.col("commentScore").isNotNull())
        .orderBy(F.desc("price"))
        .limit(200)
        .toPandas()
    )

    paid_pdf = (
        df.select(
            F.when(F.col("price") <= 0, F.lit("免费景点")).otherwise(F.lit("付费景点")).alias("label"),
            "heatScore",
            "commentScore",
        )
        .groupBy("label")
        .agg(F.count("*").alias("count"), F.avg("heatScore").alias("avgHeat"), F.avg("commentScore").alias("avgScore"))
        .toPandas()
    )

    top_pdf = (
        df.select("poiId", "poiName", "price", "marketPrice", "heatScore", "commentScore")
        .orderBy(F.desc("price"))
        .limit(10)
        .toPandas()
    )

    return {
        "forePage_price_distribution": to_records(price_pdf),
        "forePage_price_discount": to_records(discount_pdf),
        "forePage_price_vs_score": to_records(scatter_pdf),
        "forePage_price_paid_rate": to_records(paid_pdf),
        "forePage_price_top": to_records(top_pdf),
    }


def build_family_page(df: DataFrame) -> Dict[str, List[Dict[str, object]]]:
    family_mask = (
        F.lower(F.col("tagNameList")).contains("亲")
        | F.lower(F.col("shortFeatures")).contains("亲")
        | F.lower(F.col("sightCategoryInfo")).contains("亲")
        | F.lower(F.col("tagNameList")).contains("娃")
    )
    family_df = df.where(family_mask)
    city_col = "crawlDistrictName" if "crawlDistrictName" in df.columns else "districtName"

    tags_pdf = (
        family_df.select(F.explode(F.coalesce("tags_array", F.array(F.lit("")))).alias("tag"))
        .where(F.col("tag").isNotNull() & (F.col("tag") != ""))
        .groupBy("tag")
        .agg(F.count("*").alias("value"))
        .orderBy(F.desc("value"))
        .limit(80)
        .toPandas()
    )

    district_pdf = (
        family_df.groupBy(city_col)
        .agg(F.count("*").alias("count"), F.avg("commentScore").alias("avgScore"))
        .orderBy(F.desc("count"))
        .limit(15)
        .toPandas()
    )
    district_pdf.rename(columns={city_col: "district"}, inplace=True)

    level_pdf = (
        family_df.where(F.col("sightLevelStr").isNotNull())
        .groupBy("sightLevelStr")
        .agg(F.count("*").alias("count"))
        .orderBy(F.desc("count"))
        .toPandas()
    )

    heat_pdf = (
        family_df.withColumn(
            "bucket",
            F.when(F.col("heatScore") < 5, F.lit("0-4"))
            .when(F.col("heatScore") < 7, F.lit("5-6"))
            .when(F.col("heatScore") < 9, F.lit("7-8"))
            .otherwise(F.lit("9-10")),
        )
        .groupBy("bucket")
        .agg(F.count("*").alias("count"), F.avg("heatScore").alias("avgHeat"))
        .orderBy("bucket")
        .toPandas()
    )

    booking_default = F.array(
        F.struct(
            F.lit("未知").alias("name"),
            F.lit("").alias("type"),
            F.lit("").alias("code"),
        )
    )
    booking_pdf = (
        family_df.select(F.explode(F.coalesce("booking_tags", booking_default)).alias("b"))
        .select("b.name")
        .where(F.col("name").isNotNull())
        .groupBy("name")
        .count()
        .orderBy(F.desc("count"))
        .toPandas()
    )

    return {
        "forePage_family_tags": to_records(tags_pdf),
        "forePage_family_district": to_records(district_pdf),
        "forePage_family_level": to_records(level_pdf),
        "forePage_family_heat": to_records(heat_pdf),
        "forePage_family_booking": to_records(booking_pdf),
    }


def build_media_page(df: DataFrame) -> Dict[str, List[Dict[str, object]]]:
    video_pdf = (
        df.select(F.when(F.col("hasVideo") == True, F.lit("可看视频")).otherwise(F.lit("无视频")).alias("label"), "heatScore")
        .groupBy("label")
        .agg(F.count("*").alias("count"), F.avg("heatScore").alias("avgHeat"))
        .toPandas()
    )

    distance_pdf = (
        df.where(F.col("distanceKm").isNotNull())
        .withColumn(
            "bucket",
            F.when(F.col("distanceKm") <= 1, F.lit("0-1km"))
            .when(F.col("distanceKm") <= 5, F.lit("1-5km"))
            .when(F.col("distanceKm") <= 15, F.lit("5-15km"))
            .when(F.col("distanceKm") <= 30, F.lit("15-30km"))
            .otherwise(F.lit("30km+")),
        )
        .groupBy("bucket")
        .agg(F.count("*").alias("count"), F.avg("heatScore").alias("avgHeat"))
        .orderBy("bucket")
        .toPandas()
    )

    opening_pdf = (
        df.select(F.col("openStatus").alias("status"))
        .where(F.col("status").isNotNull() & (F.col("status") != ""))
        .groupBy("status")
        .count()
        .orderBy(F.desc("count"))
        .toPandas()
    )

    hot_pdf = (
        df.select(F.col("newHotInfo"))
        .where(F.col("newHotInfo").isNotNull() & (F.col("newHotInfo") != ""))
        .withColumn("info", F.from_json("newHotInfo", StructType([StructField("name", StringType(), True)])))
        .select("info.name")
        .where(F.col("name").isNotNull())
        .groupBy("name")
        .count()
        .orderBy(F.desc("count"))
        .limit(60)
        .toPandas()
    )

    price_tag_default = F.array(
        F.struct(
            F.lit("无价格标签").alias("name"),
            F.lit("").alias("type"),
            F.lit("").alias("code"),
        )
    )
    price_tag_pdf = (
        df.select(F.explode(F.coalesce("price_tags", price_tag_default)).alias("p"))
        .select("p.name")
        .where(F.col("name").isNotNull())
        .groupBy("name")
        .count()
        .orderBy(F.desc("count"))
        .toPandas()
    )

    return {
        "forePage_media_video": to_records(video_pdf),
        "forePage_media_distance": to_records(distance_pdf),
        "forePage_media_opening": to_records(opening_pdf),
        "forePage_media_event": to_records(hot_pdf),
        "forePage_media_price_tag": to_records(price_tag_pdf),
    }


SCHEMAS: Dict[str, StructType] = {
    "forePage_filter_city": StructType(
        [
            StructField("city", StringType(), True),
            StructField("attractionCount", LongType(), True),
            StructField("avgPrice", DoubleType(), True),
            StructField("avgHeat", DoubleType(), True),
            StructField("paidRate", DoubleType(), True),
        ]
    ),
    "forePage_filter_level": StructType(
        [
            StructField("sightLevelStr", StringType(), True),
            StructField("count", LongType(), True),
            StructField("avgScore", DoubleType(), True),
            StructField("avgPrice", DoubleType(), True),
        ]
    ),
    "forePage_filter_tags": StructType(
        [
            StructField("tag", StringType(), True),
            StructField("value", LongType(), True),
            StructField("avgHeat", DoubleType(), True),
        ]
    ),
    "forePage_filter_booking": StructType(
        [
            StructField("name", StringType(), True),
            StructField("count", LongType(), True),
        ]
    ),
    "forePage_filter_price": StructType(
        [
            StructField("priceBucket", StringType(), True),
            StructField("count", LongType(), True),
            StructField("avgScore", DoubleType(), True),
        ]
    ),
    "forePage_price_distribution": StructType(
        [
            StructField("bucket", StringType(), True),
            StructField("count", LongType(), True),
        ]
    ),
    "forePage_price_discount": StructType(
        [
            StructField("bucket", StringType(), True),
            StructField("count", LongType(), True),
            StructField("avgDiscount", DoubleType(), True),
        ]
    ),
    "forePage_price_vs_score": StructType(
        [
            StructField("price", DoubleType(), True),
            StructField("commentScore", DoubleType(), True),
        ]
    ),
    "forePage_price_paid_rate": StructType(
        [
            StructField("label", StringType(), True),
            StructField("count", LongType(), True),
            StructField("avgHeat", DoubleType(), True),
            StructField("avgScore", DoubleType(), True),
        ]
    ),
    "forePage_price_top": StructType(
        [
            StructField("poiId", LongType(), True),
            StructField("poiName", StringType(), True),
            StructField("price", DoubleType(), True),
            StructField("marketPrice", DoubleType(), True),
            StructField("heatScore", DoubleType(), True),
            StructField("commentScore", DoubleType(), True),
        ]
    ),
    "forePage_family_tags": StructType(
        [
            StructField("tag", StringType(), True),
            StructField("value", LongType(), True),
        ]
    ),
    "forePage_family_district": StructType(
        [
            StructField("district", StringType(), True),
            StructField("count", LongType(), True),
            StructField("avgScore", DoubleType(), True),
        ]
    ),
    "forePage_family_level": StructType(
        [
            StructField("sightLevelStr", StringType(), True),
            StructField("count", LongType(), True),
        ]
    ),
    "forePage_family_heat": StructType(
        [
            StructField("bucket", StringType(), True),
            StructField("count", LongType(), True),
            StructField("avgHeat", DoubleType(), True),
        ]
    ),
    "forePage_family_booking": StructType(
        [
            StructField("name", StringType(), True),
            StructField("count", LongType(), True),
        ]
    ),
    "forePage_media_video": StructType(
        [
            StructField("label", StringType(), True),
            StructField("count", LongType(), True),
            StructField("avgHeat", DoubleType(), True),
        ]
    ),
    "forePage_media_distance": StructType(
        [
            StructField("bucket", StringType(), True),
            StructField("count", LongType(), True),
            StructField("avgHeat", DoubleType(), True),
        ]
    ),
    "forePage_media_opening": StructType(
        [
            StructField("status", StringType(), True),
            StructField("count", LongType(), True),
        ]
    ),
    "forePage_media_event": StructType(
        [
            StructField("name", StringType(), True),
            StructField("count", LongType(), True),
        ]
    ),
    "forePage_media_price_tag": StructType(
        [
            StructField("name", StringType(), True),
            StructField("count", LongType(), True),
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
    spark = build_spark("ForePageAggregator", args.fs, args.metastore_uris)
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = prepare_dataframe(load_table(spark, args.database, args.table))
        payload = {}
        filter_tables = build_filter_page(df)
        price_tables = build_price_page(df)
        family_tables = build_family_page(df)
        media_tables = build_media_page(df)

        output_db = args.output_db
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {output_db}")
        for table_name, records in {
            **filter_tables,
            **price_tables,
            **family_tables,
            **media_tables,
        }.items():
            write_table(spark, output_db, table_name, records)

        payload["generatedAt"] = datetime.now(timezone.utc).isoformat()
        payload.update(
            {
                "filter": filter_tables,
                "price": price_tables,
                "family": family_tables,
                "media": media_tables,
            }
        )
        with output_path.open("w", encoding="utf-8") as fp:
            fp.write(str(payload))
        print("ForePage 可视化数据生成完成")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

