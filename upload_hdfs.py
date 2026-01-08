from __future__ import annotations

import argparse
import posixpath
import shutil
from pathlib import Path

from hdfs import InsecureClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="上传本地 CSV 到 HDFS 指定路径")
    parser.add_argument(
        "--namenode",
        default="node1:9870",
        help="NameNode WebHDFS 地址，格式 host:port，默认 node1:9870",
    )
    parser.add_argument(
        "--user",
        default="hdfs",
        help="HDFS 用户名，默认 hdfs",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=Path("../spider/data/attractions.csv"),
        help="本地 CSV 文件路径，默认 ../spider/data/attractions.csv",
    )
    parser.add_argument(
        "--remote",
        default="/trainData/attractions.csv",
        help="HDFS 目标路径，默认 /trainData/attractions.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标已存在则覆盖，默认追加写入（保留旧文件）",
    )
    return parser.parse_args()


def ensure_parent(client: InsecureClient, hdfs_path: str) -> None:
    parent = posixpath.dirname(hdfs_path.rstrip("/")) or "/"
    if parent != "/" and not client.status(parent, strict=False):
        client.makedirs(parent)


def upload_to_hdfs(
    local_path: Path,
    remote_path: str,
    namenode: str,
    user: str,
    overwrite: bool,
) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"本地文件不存在：{local_path}")

    client = InsecureClient(f"http://{namenode}", user=user)
    ensure_parent(client, remote_path)

    if client.status(remote_path, strict=False) and not overwrite:
        print(f"HDFS 已存在 {remote_path}，使用 --overwrite 可覆盖。")
        return

    with local_path.open("rb") as reader, client.write(
        remote_path, overwrite=True
    ) as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)

    print(f"已将 {local_path} 上传至 hdfs://{namenode}{remote_path}")


if __name__ == "__main__":
    args = parse_args()
    upload_to_hdfs(
        local_path=args.local,
        remote_path=args.remote,
        namenode=args.namenode,
        user=args.user,
        overwrite=args.overwrite,
    )

