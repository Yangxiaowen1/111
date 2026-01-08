from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf  # type: ignore
from django.conf import settings
from pyhive import hive

from user.models import (
    UserClickBehavior,
    UserCommentBehavior,
    UserDetailViewBehavior,
    UserFavoriteBehavior,
)

MODEL_DIR = Path(settings.BASE_DIR) / "recommend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "dnn_rank_ctr"
META_FILE = MODEL_DIR / "dnn_rank_meta.npz"


def _collect_behaviors() -> List[Tuple[int, int, float]]:
    """从本地 MySQL 行为表中收集 (user_id, poi_id, weight)。"""

    def _iter(qs, weight: float) -> Iterable[Tuple[int, int, float]]:
        for user_id, poi_id in qs.values_list("user_id", "poi_id"):
            if user_id and poi_id:
                yield int(user_id), int(poi_id), weight

    records: List[Tuple[int, int, float]] = []
    records.extend(_iter(UserClickBehavior.objects.all(), 1.0))
    records.extend(_iter(UserDetailViewBehavior.objects.all(), 0.8))
    records.extend(_iter(UserFavoriteBehavior.objects.filter(is_favorite=True), 1.5))
    records.extend(_iter(UserCommentBehavior.objects.all(), 2.0))
    return records


def _load_item_features() -> Dict[int, Dict[str, object]]:
    """从 Hive dwd_attractions 中加载排序需要的特征。"""
    cfg = settings.HIVE_CONFIG
    table = cfg.get("table", "dwd_attractions")
    sql = f"""
        SELECT poiId, districtName, price, heatScore, commentScore, tagNameList
        FROM {table}
    """
    connection = hive.Connection(
        host=cfg["host"],
        port=cfg["port"],
        username=cfg.get("username"),
        password=cfg.get("password") or None,
        database=cfg.get("database", "default"),
    )
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    feature_map: Dict[int, Dict[str, object]] = {}
    for row in rows:
        record = dict(zip(columns, row))
        poi_id = int(record["poiId"])
        feature_map[poi_id] = record
    return feature_map


def _build_mappings(records: Sequence[Tuple[int, int, float]]):
    user_ids = sorted({u for u, _, _ in records})
    poi_ids = sorted({i for _, i, _ in records})
    user_to_idx = {u: idx for idx, u in enumerate(user_ids)}
    poi_to_idx = {i: idx for idx, i in enumerate(poi_ids)}
    return user_to_idx, poi_to_idx, np.array(user_ids), np.array(poi_ids)


def _negative_samples(
    positives: Sequence[Tuple[int, int, float]],
    num_users: int,
    num_items: int,
    ratio: int = 1,
) -> List[Tuple[int, int]]:
    """简单负采样：随机生成 user-item 组合，过滤掉已存在的正样本。"""
    pos_set = {(u, i) for u, i, _ in positives}
    rng = np.random.default_rng(2024)
    samples: List[Tuple[int, int]] = []
    for _ in range(len(positives) * ratio):
        u = int(rng.integers(0, num_users))
        i = int(rng.integers(0, num_items))
        samples.append((u, i))
    return [s for s in samples if s not in pos_set]


@dataclass
class DNNRankConfig:
    embedding_dim: int = 32
    hidden_units: Sequence[int] = (64, 32)
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    neg_ratio: int = 2


class DNNRankTrainer:
    """DNN CTR Rank 训练器：User + POI Embedding + 多特征全连接。"""

    def __init__(self, config: DNNRankConfig | None = None):
        self.config = config or DNNRankConfig()

    def train(self) -> None:
        records = _collect_behaviors()
        if not records:
            raise ValueError("缺少可用的用户行为数据，无法训练排序模型")

        feature_map = _load_item_features()
        # 过滤掉 Hive 中不存在的 poi
        records = [r for r in records if r[1] in feature_map]
        if not records:
            raise ValueError("行为数据中的景点在 Hive 中均不存在，请检查数据是否同步")

        user_to_idx, poi_to_idx, user_ids, poi_ids = _build_mappings(records)
        num_users = len(user_ids)
        num_items = len(poi_ids)

        # 正样本
        user_indices = np.array([user_to_idx[u] for u, _, _ in records], dtype=np.int32)
        item_indices = np.array([poi_to_idx[i] for _, i, _ in records], dtype=np.int32)
        labels = np.ones(len(records), dtype=np.float32)

        # 负样本
        negatives = _negative_samples(records, num_users, num_items, ratio=self.config.neg_ratio)
        if negatives:
            neg_users, neg_items = zip(*negatives)
            user_indices = np.concatenate([user_indices, np.array(neg_users, dtype=np.int32)])
            item_indices = np.concatenate([item_indices, np.array(neg_items, dtype=np.int32)])
            labels = np.concatenate([labels, np.zeros(len(negatives), dtype=np.float32)])

        # 构造 POI 特征映射到索引空间
        district_names = sorted(
            {
                (feature_map[int(p)]["districtName"] or "未知")
                for p in poi_ids
                if p in feature_map
            }
        )
        district_to_idx = {name: idx for idx, name in enumerate(district_names)}

        def _build_poi_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            price_arr = np.zeros(num_items, dtype=np.float32)
            heat_arr = np.zeros(num_items, dtype=np.float32)
            score_arr = np.zeros(num_items, dtype=np.float32)
            district_arr = np.zeros(num_items, dtype=np.int32)
            for local_idx, poi_id in enumerate(poi_ids):
                feat = feature_map[int(poi_id)]
                price = float(feat.get("price") or 0.0)
                heat = float(feat.get("heatScore") or 0.0)
                score = float(feat.get("commentScore") or 0.0)
                district = str(feat.get("districtName") or "未知")
                price_arr[local_idx] = price
                heat_arr[local_idx] = heat
                score_arr[local_idx] = score
                district_arr[local_idx] = district_to_idx.get(district, 0)
            # 简单归一化
            price_arr = (price_arr - price_arr.mean()) / (price_arr.std() + 1e-6)
            heat_arr = (heat_arr - heat_arr.mean()) / (heat_arr.std() + 1e-6)
            score_arr = (score_arr - score_arr.mean()) / (score_arr.std() + 1e-6)
            return price_arr, heat_arr, score_arr, district_arr

        price_arr, heat_arr, score_arr, district_arr = _build_poi_features()

        # 为每条训练样本查表取特征
        price_feat = price_arr[item_indices]
        heat_feat = heat_arr[item_indices]
        score_feat = score_arr[item_indices]
        district_feat = district_arr[item_indices]

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "user_id": user_indices,
                    "poi_id": item_indices,
                    "price": price_feat,
                    "heat": heat_feat,
                    "score": score_feat,
                    "district_id": district_feat,
                },
                labels,
            )
        )
        dataset = dataset.shuffle(buffer_size=len(labels)).batch(self.config.batch_size)

        model = self._build_model(num_users, num_items, len(district_names))
        model.fit(dataset, epochs=self.config.epochs, verbose=1)

        # 保存模型和元数据
        model.save(MODEL_PATH, include_optimizer=False)
        np.savez(
            META_FILE,
            user_ids=user_ids,
            poi_ids=poi_ids,
            price_arr=price_arr,
            heat_arr=heat_arr,
            score_arr=score_arr,
            district_arr=district_arr,
            district_names=np.array(district_names, dtype=object),
        )

        print(f"已保存排序模型到 {MODEL_PATH}，元数据到 {META_FILE}")

    def _build_model(self, num_users: int, num_items: int, num_districts: int) -> tf.keras.Model:
        cfg = self.config
        user_input = tf.keras.Input(shape=(), name="user_id", dtype=tf.int32)
        poi_input = tf.keras.Input(shape=(), name="poi_id", dtype=tf.int32)
        district_input = tf.keras.Input(shape=(), name="district_id", dtype=tf.int32)

        price_input = tf.keras.Input(shape=(), name="price", dtype=tf.float32)
        heat_input = tf.keras.Input(shape=(), name="heat", dtype=tf.float32)
        score_input = tf.keras.Input(shape=(), name="score", dtype=tf.float32)

        user_emb = tf.keras.layers.Embedding(num_users, cfg.embedding_dim, name="user_embedding")(user_input)
        poi_emb = tf.keras.layers.Embedding(num_items, cfg.embedding_dim, name="poi_embedding")(poi_input)
        dist_emb = tf.keras.layers.Embedding(num_districts or 1, cfg.embedding_dim // 2, name="district_embedding")(
            district_input
        )

        user_vec = tf.keras.layers.Flatten()(user_emb)
        poi_vec = tf.keras.layers.Flatten()(poi_emb)
        dist_vec = tf.keras.layers.Flatten()(dist_emb)

        # 将标量特征扩展为形状 (None, 1)，以便与 embedding 向量在同一维度上拼接
        price_vec = tf.keras.layers.Reshape((1,))(price_input)
        heat_vec = tf.keras.layers.Reshape((1,))(heat_input)
        score_vec = tf.keras.layers.Reshape((1,))(score_input)

        concat = tf.keras.layers.Concatenate()(
            [user_vec, poi_vec, dist_vec, price_vec, heat_vec, score_vec]
        )
        x = concat
        for units in cfg.hidden_units:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        model = tf.keras.Model(
            inputs={
                "user_id": user_input,
                "poi_id": poi_input,
                "district_id": district_input,
                "price": price_input,
                "heat": heat_input,
                "score": score_input,
            },
            outputs=output,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model


class DNNRankRecommender:
    """在线排序推荐器：给定用户 ID，对 Hive 景点进行 CTR 排序。"""

    def __init__(self, model_path: Path = MODEL_PATH, meta_file: Path = META_FILE):
        if not model_path.exists() or not meta_file.exists():
            raise FileNotFoundError("未找到排序模型，请先运行 train_dnn_rank 命令")
        self.model = tf.keras.models.load_model(model_path)
        payload = np.load(meta_file, allow_pickle=True)
        self.user_ids = payload["user_ids"]
        self.poi_ids = payload["poi_ids"]
        self.price_arr = payload["price_arr"]
        self.heat_arr = payload["heat_arr"]
        self.score_arr = payload["score_arr"]
        self.district_arr = payload["district_arr"]
        self.user_index = {int(u): idx for idx, u in enumerate(self.user_ids)}

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        if user_id not in self.user_index:
            # 冷启动：直接根据热度排序
            order = np.argsort(-self.heat_arr)
            top = order[:top_k]
            return [int(self.poi_ids[i]) for i in top]

        u_idx = self.user_index[user_id]
        num_items = len(self.poi_ids)
        user_ids = np.full(shape=(num_items,), fill_value=u_idx, dtype=np.int32)
        poi_indices = np.arange(num_items, dtype=np.int32)

        inputs = {
            "user_id": user_ids,
            "poi_id": poi_indices,
            "district_id": self.district_arr.astype(np.int32),
            "price": self.price_arr.astype(np.float32),
            "heat": self.heat_arr.astype(np.float32),
            "score": self.score_arr.astype(np.float32),
        }

        scores = self.model.predict(inputs, batch_size=256, verbose=0).reshape(-1)
        top_indices = np.argpartition(-scores, range(min(top_k, len(scores))))[:top_k]
        ranked = sorted(top_indices, key=lambda idx: scores[idx], reverse=True)
        return [int(self.poi_ids[idx]) for idx in ranked]


