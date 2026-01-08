from __future__ import annotations

import argparse
from typing import Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DWD 层景点数据清洗脚本")
    parser.add_argument(
        "--fs",
        default="hdfs://node1:8020",
        help="HDFS 默认文件系统（fs.defaultFS），默认 hdfs://node1:8020",
    )
    parser.add_argument(
        "--metastore_uris",
        default='thrift://node1:9083',
        help="Hive Metastore URIs（如 thrift://node1:9083），默认为空使用本地 metastore",
    )
    parser.add_argument(
        "--input_db",
        default="default",
        help="ODS 所在数据库，默认 default",
    )
    parser.add_argument(
        "--input_table",
        default="ODS_attractions",
        help="ODS 表名，默认 ODS_attractions",
    )
    parser.add_argument(
        "--output_db",
        default="default",
        help="default 目标数据库，默认 default",
    )
    parser.add_argument(
        "--output_table",
        default="DWD_attractions",
        help="DWD 目标表名，默认 DWD_attractions",
    )
    parser.add_argument(
        "--mode",
        choices=("overwrite", "append"),
        default="overwrite",
        help="写入模式，默认 overwrite",
    )
    parser.add_argument(
        "--repartition",
        type=int,
        default=None,
        help="清洗后重分区数量，用于控制写入并行度",
    )
    return parser.parse_args()


def build_spark(app_name: str, fs: str, metastore_uris: Optional[str]) -> SparkSession:
    builder = SparkSession.builder.appName(app_name).enableHiveSupport()
    builder = builder.config("spark.hadoop.fs.defaultFS", fs)
    if metastore_uris:
        builder = builder.config("hive.metastore.uris", metastore_uris)
    return builder.getOrCreate()


def load_ods(spark: SparkSession, db: str, table: str) -> DataFrame:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")
    return spark.table(f"{db}.{table}")


def normalize_strings(df: DataFrame) -> DataFrame:
    updated = df
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            updated = updated.withColumn(field.name, F.trim(F.col(field.name)))
    return updated


def cast_numeric(df: DataFrame) -> DataFrame:
    numeric_mappings = {
        "poiId": "bigint",
        "commentCount": "bigint",
        "price": "double",
        "marketPrice": "double",
        "marketDiscountPrice": "double",
        "underlineDiscountPrice": "double",
        "commentScore": "double",
        "heatScore": "double",
        "latitude": "double",
        "longitude": "double",
        "timestamp": "bigint",
    }
    cleaned = df
    for column, target_type in numeric_mappings.items():
        if column in df.columns:
            cleaned = cleaned.withColumn(
                column,
                F.col(column).cast(target_type),
            )
    return cleaned


def deduplicate(df: DataFrame) -> DataFrame:
    if "poiId" not in df.columns:
        return df
    ts_col = "timestamp" if "timestamp" in df.columns else None
    window = (
        Window.partitionBy("poiId").orderBy(
            F.col(ts_col).desc_nulls_last() if ts_col else F.lit(1)
        )
    )
    ranked = df.withColumn("rn", F.row_number().over(window))
    return ranked.filter(F.col("rn") == 1).drop("rn")


def enrich_fields(df: DataFrame) -> DataFrame:
    """补充一些衍生字段，如价格折扣率等，可按需扩展。"""
    if {"price", "marketPrice"}.issubset(df.columns):
        df = df.withColumn(
            "discount_rate",
            F.when(
                (F.col("marketPrice") > 0) & F.col("price").isNotNull(),
                F.col("price") / F.col("marketPrice"),
            ).otherwise(None),
        )
    return df


def clean_data(df: DataFrame) -> DataFrame:
    cleaned = normalize_strings(df)
    cleaned = cast_numeric(cleaned)
    cleaned = cleaned.filter(
        F.col("poiId").isNotNull() & F.col("poiName").isNotNull()
    )
    cleaned = deduplicate(cleaned)
    cleaned = enrich_fields(cleaned)
    return cleaned


def write_dwd(
    df: DataFrame,
    db: str,
    table: str,
    mode: str,
    repartition: Optional[int],
) -> None:
    spark = df.sparkSession
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")
    final_df = df.repartition(repartition) if repartition else df
    (
        final_df.write.mode(mode)
        .format("hive")
        .saveAsTable(f"{db}.{table}")
    )


def main() -> None:
    args = parse_args()
    spark = build_spark("DWD_attractions_cleaner", args.fs, args.metastore_uris)
    try:
        ods_df = load_ods(spark, args.input_db, args.input_table)
        dwd_df = clean_data(ods_df)
        write_dwd(
            df=dwd_df,
            db=args.output_db,
            table=args.output_table,
            mode=args.mode,
            repartition=args.repartition,
        )
        print(
            f"成功写入 Hive 表 {args.output_db}.{args.output_table} "
            f"(mode={args.mode}, rows={dwd_df.count()})"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

