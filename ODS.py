from __future__ import annotations

import argparse
from typing import Optional

from pyspark.sql import DataFrame, SparkSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ODS 层景点数据入湖脚本")
    parser.add_argument(
        "--fs",
        default="hdfs://node1:8020",
        help="HDFS 默认文件系统（fs.defaultFS），默认 hdfs://node1:8020",
    )
    parser.add_argument(
        "--input",
        default="/trainData/attractions.csv",
        help="HDFS 上的 CSV 输入路径（相对 fs.defaultFS），默认 /trainData/attractions.csv",
    )
    parser.add_argument(
        "--database",
        default="default",
        help="目标 Hive 数据库，默认 default",
    )
    parser.add_argument(
        "--table",
        default="ODS_attractions",
        help="目标 Hive 表名，默认 ODS_attractions",
    )
    parser.add_argument(
        "--mode",
        choices=("overwrite", "append"),
        default="overwrite",
        help="写入模式，默认 overwrite",
    )
    parser.add_argument(
        "--metastore_uris",
        default='thrift://node1:9083',
        help="可选 Hive Metastore URIs（如 thrift://node1:9083）",
    )
    parser.add_argument(
        "--repartition",
        type=int,
        default=None,
        help="可选重分区数量，控制写入并行度",
    )
    return parser.parse_args()


def build_spark(app_name: str, metastore_uris: Optional[str], default_fs: str) -> SparkSession:
    builder = SparkSession.builder.appName(app_name).enableHiveSupport()
    if metastore_uris:
        builder = builder.config("hive.metastore.uris", metastore_uris)
    builder = builder.config("spark.hadoop.fs.defaultFS", default_fs)
    return builder.getOrCreate()


def read_source(spark: SparkSession, input_path: str) -> DataFrame:
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("escape", "\"")
        .csv(input_path)
    )


def write_to_hive(
    df: DataFrame,
    database: str,
    table: str,
    mode: str,
    repartition: Optional[int],
) -> None:
    spark = df.sparkSession
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
    target = f"{database}.{table}"
    final_df = df.repartition(repartition) if repartition else df
    (
        final_df.write.mode(mode)
        .format("hive")
        .saveAsTable(target)
    )


def main() -> None:
    args = parse_args()
    spark = build_spark("ODS_attractions_loader", args.metastore_uris, args.fs)
    try:
        source_df = read_source(spark, args.input)
        write_to_hive(
            df=source_df,
            database=args.database,
            table=args.table,
            mode=args.mode,
            repartition=args.repartition,
        )
        print(
            f"成功写入 Hive 表 {args.database}.{args.table} "
            f"(mode={args.mode}, rows={source_df.count()})"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

