from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)


DIRECT_CITY_SUFFIX = {
    "北京": "北京市",
    "上海": "上海市",
    "天津": "天津市",
    "重庆": "重庆市",
}

SPECIAL_REGION_SUFFIX = {
    "内蒙古": "内蒙古自治区",
    "广西": "广西壮族自治区",
    "宁夏": "宁夏回族自治区",
    "新疆": "新疆维吾尔自治区",
    "西藏": "西藏自治区",
    "香港": "香港特别行政区",
    "澳门": "澳门特别行政区",
    "台湾": "台湾省",
}

# 城市到省份的映射（去除"市"后缀后匹配）
CITY_TO_PROVINCE = {
    "长沙": "湖南省",
    "厦门": "福建省",
    "西安": "陕西省",
    "银川": "宁夏回族自治区",
    "桂林": "广西壮族自治区",
    "大理": "云南省",
    "武汉": "湖北省",
    "长春": "吉林省",
    "哈尔滨": "黑龙江省",
    "杭州": "浙江省",
    "南京": "江苏省",
    "成都": "四川省",
    "广州": "广东省",
    "深圳": "广东省",
    "苏州": "江苏省",
    "青岛": "山东省",
    "大连": "辽宁省",
    "沈阳": "辽宁省",
    "济南": "山东省",
    "郑州": "河南省",
    "石家庄": "河北省",
    "太原": "山西省",
    "合肥": "安徽省",
    "南昌": "江西省",
    "福州": "福建省",
    "昆明": "云南省",
    "贵阳": "贵州省",
    "海口": "海南省",
    "兰州": "甘肃省",
    "西宁": "青海省",
    "乌鲁木齐": "新疆维吾尔自治区",
    "拉萨": "西藏自治区",
    "呼和浩特": "内蒙古自治区",
    "南宁": "广西壮族自治区",
}

IMAGE_COLUMNS = [
    "coverImageUrl",
    "dynamicCoverImageUrl",
    "coverImage",
    "imageUrl",
    "imgUrl",
    "image",
    "firstImage",
    "frontImage",
    "mainImage",
]
IMAGE_JSON_COLUMNS = [
    "imageList",
    "images",
    "imageUrls",
    "coverList",
    "album",
]
DETAIL_COLUMNS = ["detailUrl", "url", "wapUrl", "jumpUrl", "detailLink"]
CITY_COLUMNS = ["districtName", "crawlDistrictName", "cityName", "provinceName"]
DESC_COLUMNS = ["shortDesc", "absDesc", "description", "tips", "summary", "intro"]
PRICE_COLUMNS = ["price", "marketPrice", "marketDiscountPrice", "discountPrice"]


def normalize_region_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    name = str(name).strip()
    # 1. 处理直辖市（北京、上海、天津、重庆）→ 加"市"
    if name in DIRECT_CITY_SUFFIX:
        return DIRECT_CITY_SUFFIX[name]
    # 2. 处理特殊行政区（内蒙古、广西等）
    if name in SPECIAL_REGION_SUFFIX:
        return SPECIAL_REGION_SUFFIX[name]
    # 3. 如果已经是完整的省/自治区名称，直接返回
    if name.endswith(("省", "自治区", "特别行政区")):
        return name
    # 4. 处理城市名（带"市"或不带"市"）→ 映射到对应省份
    city_name = name[:-1] if name.endswith("市") else name
    if city_name in CITY_TO_PROVINCE:
        return CITY_TO_PROVINCE[city_name]
    # 5. 处理州、盟等特殊情况，保持原样
    if name.endswith(("州", "盟")):
        return name
    # 6. 其他情况，如果已经是省份名称格式，直接返回；否则默认加"省"
    if name.endswith("省"):
        return name
    return f"{name}省"


def pick_column(df: DataFrame, candidates: List[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def optional_expr(df: DataFrame, alias: str, candidates: List[str], default=None):
    column = pick_column(df, candidates)
    if column:
        return F.col(column).alias(alias)
    return F.lit(default).alias(alias)


def parse_image_value(value) -> Optional[str]:
    """尝试从任意结构中提取图片 URL。"""
    if not value:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("http"):
            return stripped
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
    else:
        parsed = value
    if isinstance(parsed, list):
        for item in parsed:
            url = parse_image_value(item)
            if url:
                return url
        return None
    if isinstance(parsed, dict):
        for key in ("imageUrl", "url", "coverUrl", "imgUrl", "originImage", "smallImage", "bigImage"):
            nested = parsed.get(key)
            if isinstance(nested, str) and nested.startswith("http"):
                return nested
        for nested in parsed.values():
            url = parse_image_value(nested)
            if url:
                return url
    if isinstance(parsed, str) and parsed.startswith("http"):
        return parsed
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="大屏数据分析聚合脚本")
    parser.add_argument(
        "--fs",
        default="hdfs://node1:8020",
        help="HDFS 默认文件系统（fs.defaultFS），默认 hdfs://node1:8020",
    )
    parser.add_argument(
        "--metastore_uris",
        default="thrift://node1:9083",
        help="Hive Metastore URIs，默认 thrift://node1:9083",
    )
    parser.add_argument(
        "--database",
        default="default",
        help="读取 Hive 数据库，默认 default",
    )
    parser.add_argument(
        "--table",
        default="dwd_attractions",
        help="读取 Hive 表名，默认 dwd_attractions",
    )
    parser.add_argument(
        "--output_db",
        default="default",
        help="写入 Hive 数据库，默认 default",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spark/outputs/dashboard_summary.json"),
        help="可选：同时生成 JSON 文件，默认 spark/outputs/dashboard_summary.json",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="榜单/表格展示的前 N 条数据，默认 10",
    )
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


def ensure_column(df: DataFrame, name: str, default_expr) -> DataFrame:
    if name in df.columns:
        return df
    return df.withColumn(name, default_expr if isinstance(default_expr, F.Column) else F.lit(default_expr))


def build_cards(df: DataFrame) -> Dict[str, float]:
    total_attractions = df.select("poiId").distinct().count() if "poiId" in df.columns else df.count()
    avg_score = df.agg(F.avg(F.col("commentScore"))).first()[0] if "commentScore" in df.columns else None
    total_comments = df.agg(F.sum(F.col("commentCount"))).first()[0] if "commentCount" in df.columns else None
    premium_cnt = (
        df.filter(F.col("commentScore") >= 4.5).count()
        if "commentScore" in df.columns
        else 0
    )
    return {
        "totalAttractions": int(total_attractions or 0),
        "avgScore": round(avg_score, 2) if avg_score is not None else None,
        "totalComments": int(total_comments or 0) if total_comments else 0,
        "highScorePlaces": int(premium_cnt),
    }


def build_map(df: DataFrame) -> List[Dict[str, float]]:
    target_col = "provinceName" if "provinceName" in df.columns else None
    if not target_col and "crawlDistrictName" in df.columns:
        target_col = "crawlDistrictName"
    if not target_col:
        return []
    spark_df = (
        df.groupBy(F.col(target_col).alias("name"))
        .agg(F.countDistinct("poiId").alias("value"))
        .orderBy(F.desc("value"))
        .limit(34)
    )
    return [
        {"name": normalize_region_name(row["name"]), "value": int(row["value"] or 0)}
        for row in spark_df.collect()
    ]


def build_wordcloud(df: DataFrame, limit: int) -> List[Dict[str, float]]:
    score_col = "heatScore" if "heatScore" in df.columns else (
        "commentCount" if "commentCount" in df.columns else None
    )
    if not score_col or "poiName" not in df.columns:
        return []
    rows = (
        df.select("poiName", F.col(score_col).alias("value"))
        .orderBy(F.desc("value"))
        .limit(limit)
        .collect()
    )
    return [{"name": row["poiName"], "value": float(row["value"] or 0)} for row in rows]


def build_trend(df: DataFrame) -> List[Dict[str, float]]:
    if "timestamp" not in df.columns:
        return []
    trend_df = (
        df.withColumn("event_date", F.from_unixtime(F.col("timestamp") / 1000).cast("date"))
        .groupBy("event_date")
        .agg(F.avg("commentScore").alias("avgScore"))
        .orderBy("event_date")
        .limit(30)
    )
    rows = trend_df.collect()
    return [
        {"date": row["event_date"].isoformat(), "avgScore": round(row["avgScore"], 2)}
        for row in rows
        if row["event_date"] is not None
    ]


def build_category(df: DataFrame) -> List[Dict[str, float]]:
    target_col = None
    for candidate in ("categoryName", "categoryType", "tagName"):
        if candidate in df.columns:
            target_col = candidate
            break
    if not target_col:
        return []
    agg_df = (
        df.groupBy(F.col(target_col).alias("name"))
        .agg(F.countDistinct("poiId").alias("value"))
        .orderBy(F.desc("value"))
        .limit(8)
    )
    return [{"name": row["name"], "value": int(row["value"] or 0)} for row in agg_df.collect()]


def build_score_distribution(df: DataFrame) -> List[Dict[str, float]]:
    if "commentScore" not in df.columns:
        return []
    buckets = [0, 1, 2, 3, 4, 5, 6]
    bucket_df = (
        df.withColumn(
            "bucket",
            F.when(F.col("commentScore") >= 5, F.lit("5分+")).otherwise(
                F.concat(
                    F.floor(F.col("commentScore")).cast("int"),
                    F.lit("-"),
                    (F.floor(F.col("commentScore")) + 1).cast("int"),
                )
            ),
        )
        .groupBy("bucket")
        .agg(F.count("*").alias("value"))
    )
    return [{"bucket": row["bucket"], "value": int(row["value"] or 0)} for row in bucket_df.collect()]


def build_price_range(df: DataFrame) -> List[Dict[str, float]]:
    if "price" not in df.columns:
        return []
    bucket_expr = F.when(F.col("price") < 50, F.lit("0-49")) \
        .when(F.col("price") < 100, F.lit("50-99")) \
        .when(F.col("price") < 200, F.lit("100-199")) \
        .when(F.col("price") < 500, F.lit("200-499")) \
        .otherwise(F.lit("500+"))
    rows = (
        df.withColumn("priceBucket", bucket_expr)
        .groupBy("priceBucket")
        .agg(F.count("*").alias("value"))
        .orderBy("priceBucket")
        .collect()
    )
    return [{"priceBucket": row["priceBucket"], "value": int(row["value"] or 0)} for row in rows]


def build_table(df: DataFrame, limit: int) -> List[Dict[str, float]]:
    if "poiName" not in df.columns:
        return []

    select_exprs = [
        F.col("poiName").alias("poiName"),
        optional_expr(df, "commentScore", ["commentScore"]),
        optional_expr(df, "commentCount", ["commentCount"]),
        optional_expr(df, "price", PRICE_COLUMNS),
        optional_expr(df, "detailUrl", DETAIL_COLUMNS),
        optional_expr(df, "city", CITY_COLUMNS),
        optional_expr(df, "description", DESC_COLUMNS),
        F.coalesce(
            *[F.col(col) for col in IMAGE_COLUMNS if col in df.columns],
            F.lit(None),
        ).alias("primaryImage"),
        F.coalesce(
            *[F.col(col) for col in IMAGE_JSON_COLUMNS if col in df.columns],
            F.lit(None),
        ).alias("primaryImageJson"),
    ]

    table_df = df.select(*select_exprs)
    table_df = table_df.orderBy(F.col("commentScore").desc_nulls_last())
    rows = table_df.limit(limit).collect()
    result: List[Dict[str, object]] = []
    for row in rows:
        data = row.asDict(recursive=False)
        image_url = data.get("primaryImage")
        if not image_url:
            image_url = parse_image_value(data.get("primaryImageJson"))
        result.append(
            {
                "poiName": data.get("poiName"),
                "commentScore": float(data["commentScore"]) if data.get("commentScore") is not None else None,
                "commentCount": int(data["commentCount"]) if data.get("commentCount") is not None else None,
                "price": float(data["price"]) if data.get("price") is not None else None,
                "imageUrl": image_url,
                "detailUrl": data.get("detailUrl"),
                "city": data.get("city"),
                "description": data.get("description"),
            }
        )
    return result


def build_recommendations(df: DataFrame, limit: int) -> List[Dict[str, float]]:
    if "heatScore" not in df.columns or "poiName" not in df.columns:
        return []
    select_exprs = [
        F.col("poiId").alias("poiId"),
        F.col("poiName").alias("poiName"),
        optional_expr(df, "score", ["heatScore", "commentScore"]),
        optional_expr(df, "detailUrl", DETAIL_COLUMNS),
        optional_expr(df, "city", CITY_COLUMNS),
        optional_expr(df, "price", PRICE_COLUMNS),
        optional_expr(df, "description", DESC_COLUMNS),
        F.coalesce(
            *[F.col(col) for col in IMAGE_COLUMNS if col in df.columns],
            F.lit(None),
        ).alias("primaryImage"),
        F.coalesce(
            *[F.col(col) for col in IMAGE_JSON_COLUMNS if col in df.columns],
            F.lit(None),
        ).alias("primaryImageJson"),
    ]
    rows = (
        df.select(*select_exprs)
        .orderBy(F.col("score").desc_nulls_last())
        .limit(limit)
        .collect()
    )
    results: List[Dict[str, object]] = []
    for row in rows:
        if row["poiId"] is None:
            continue
        primary = row["primaryImage"]
        if not primary:
            primary = parse_image_value(row["primaryImageJson"])
        results.append(
            {
                "poiId": int(row["poiId"]),
                "name": row["poiName"],
                "score": round(row["score"], 2) if row["score"] is not None else None,
                "imageUrl": primary,
                "detailUrl": row["detailUrl"],
                "city": row["city"],
                "price": float(row["price"]) if row["price"] is not None else None,
                "description": row["description"],
            }
        )
    return results


SCHEMAS = {
    "dashaboldBig_cards": StructType(
        [
            StructField("totalAttractions", LongType(), True),
            StructField("avgScore", DoubleType(), True),
            StructField("totalComments", LongType(), True),
            StructField("highScorePlaces", LongType(), True),
        ]
    ),
    "dashaboldBig_map_heat": StructType(
        [
            StructField("name", StringType(), True),
            StructField("value", LongType(), True),
        ]
    ),
    "dashaboldBig_word_cloud": StructType(
        [
            StructField("name", StringType(), True),
            StructField("value", DoubleType(), True),
        ]
    ),
    "dashaboldBig_score_trend": StructType(
        [
            StructField("date", StringType(), True),
            StructField("avgScore", DoubleType(), True),
        ]
    ),
    "dashaboldBig_category_breakdown": StructType(
        [
            StructField("name", StringType(), True),
            StructField("value", LongType(), True),
        ]
    ),
    "dashaboldBig_score_distribution": StructType(
        [
            StructField("bucket", StringType(), True),
            StructField("value", LongType(), True),
        ]
    ),
    "dashaboldBig_price_range": StructType(
        [
            StructField("priceBucket", StringType(), True),
            StructField("value", LongType(), True),
        ]
    ),
    "dashaboldBig_top_table": StructType(
        [
            StructField("poiName", StringType(), True),
            StructField("commentScore", DoubleType(), True),
            StructField("commentCount", LongType(), True),
            StructField("price", DoubleType(), True),
            StructField("city", StringType(), True),
            StructField("description", StringType(), True),
            StructField("imageUrl", StringType(), True),
            StructField("detailUrl", StringType(), True),
        ]
    ),
    "dashaboldBig_recommendations": StructType(
        [
            StructField("poiId", LongType(), True),
            StructField("name", StringType(), True),
            StructField("score", DoubleType(), True),
            StructField("imageUrl", StringType(), True),
            StructField("detailUrl", StringType(), True),
            StructField("city", StringType(), True),
            StructField("price", DoubleType(), True),
            StructField("description", StringType(), True),
        ]
    ),
}


def write_table(spark: SparkSession, db: str, table: str, records: List[Dict[str, object]]) -> None:
    schema = SCHEMAS[table]
    if records:
        df = spark.createDataFrame(records, schema=schema)
    else:
        empty_rdd = spark.sparkContext.emptyRDD()
        df = spark.createDataFrame(empty_rdd, schema)
    df.write.mode("overwrite").saveAsTable(f"{db}.{table}")


def main() -> None:
    args = parse_args()
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spark = build_spark("DashboardBigScreenAggregator", args.fs, args.metastore_uris)
    try:
        df = load_table(spark, args.database, args.table)
        df = ensure_column(df, "commentScore", F.lit(None))
        cards = build_cards(df)
        map_heat = build_map(df)
        word_cloud = build_wordcloud(df, limit=60)
        score_trend = build_trend(df)
        category = build_category(df)
        score_distribution = build_score_distribution(df)
        price_range = build_price_range(df)
        top_table = build_table(df, args.topn)
        recommendations = build_recommendations(df, limit=6)

        summary = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "cards": cards,
            "mapHeat": map_heat,
            "wordCloud": word_cloud,
            "scoreTrend": score_trend,
            "categoryBreakdown": category,
            "scoreDistribution": score_distribution,
            "priceRange": price_range,
            "topTable": top_table,
            "recommendations": recommendations,
        }

        output_db = args.output_db
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {output_db}")
        write_table(spark, output_db, "dashaboldBig_cards", [cards])
        write_table(spark, output_db, "dashaboldBig_map_heat", map_heat)
        write_table(spark, output_db, "dashaboldBig_word_cloud", word_cloud)
        write_table(spark, output_db, "dashaboldBig_score_trend", score_trend)
        write_table(spark, output_db, "dashaboldBig_category_breakdown", category)
        write_table(spark, output_db, "dashaboldBig_score_distribution", score_distribution)
        write_table(spark, output_db, "dashaboldBig_price_range", price_range)
        write_table(spark, output_db, "dashaboldBig_top_table", top_table)
        write_table(spark, output_db, "dashaboldBig_recommendations", recommendations)

        if args.output:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as fp:
                json.dump(summary, fp, ensure_ascii=False, indent=2)
            print(f"已写入 JSON：{output_path}")
        print(f"已将大屏数据写入 Hive 数据库 {output_db}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()


