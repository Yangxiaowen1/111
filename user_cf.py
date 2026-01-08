from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import tensorflow as tf  # type: ignore
from django.conf import settings
from django.db.models import QuerySet

from user.models import (
    UserClickBehavior,
    UserCommentBehavior,
    UserDetailViewBehavior,
    UserFavoriteBehavior,
)

MODEL_DIR = Path(settings.BASE_DIR) / "recommend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_DIR / "user_cf_embeddings.npz"


def _collect_behaviors(qs: QuerySet, weight: float) -> Iterable[tuple[int, int, float]]:
    for user_id, poi_id in qs.values_list("user_id", "poi_id"):
        if user_id and poi_id:
            yield int(user_id), int(poi_id), weight


def load_interactions() -> List[tuple[int, int, float]]:
    interactions: List[tuple[int, int, float]] = []
    interactions.extend(_collect_behaviors(UserClickBehavior.objects.all(), weight=1.0))
    interactions.extend(_collect_behaviors(UserDetailViewBehavior.objects.all(), weight=0.8))
    interactions.extend(_collect_behaviors(UserFavoriteBehavior.objects.filter(is_favorite=True), weight=1.5))
    interactions.extend(_collect_behaviors(UserCommentBehavior.objects.all(), weight=2.0))
    return interactions


def build_mappings(records: Sequence[tuple[int, int, float]]):
    user_ids = sorted({user_id for user_id, _, _ in records})
    poi_ids = sorted({poi_id for _, poi_id, _ in records})
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    poi_to_idx = {poi_id: idx for idx, poi_id in enumerate(poi_ids)}
    return user_to_idx, poi_to_idx, np.array(user_ids), np.array(poi_ids)


def negative_samples(records: Sequence[tuple[int, int, float]], num_users: int, num_items: int, ratio: int = 1):
    positives = {(u, i) for u, i, _ in records}
    samples = []
    rng = np.random.default_rng(42)
    for _ in range(len(records) * ratio):
        user_idx = rng.integers(0, num_users)
        item_idx = rng.integers(0, num_items)
        samples.append((user_idx, item_idx))
    return [s for s in samples if s not in positives]


@dataclass
class TrainingConfig:
    embedding_dim: int = 32
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3


class UserCFTrainer:
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()

    def train(self) -> None:
        records = load_interactions()
        if not records:
            raise ValueError("缺少可用的用户行为数据")

        user_to_idx, poi_to_idx, user_ids, poi_ids = build_mappings(records)
        user_indices = np.array([user_to_idx[u] for u, _, _ in records], dtype=np.int32)
        item_indices = np.array([poi_to_idx[i] for _, i, _ in records], dtype=np.int32)
        labels = np.ones(len(records), dtype=np.float32)

        negatives = negative_samples(records, len(user_ids), len(poi_ids))
        if negatives:
            neg_users, neg_items = zip(*negatives)
            user_indices = np.concatenate([user_indices, np.array(neg_users, dtype=np.int32)])
            item_indices = np.concatenate([item_indices, np.array(neg_items, dtype=np.int32)])
            labels = np.concatenate([labels, np.zeros(len(negatives), dtype=np.float32)])

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "user": user_indices,
                    "item": item_indices,
                },
                labels,
            )
        )
        dataset = dataset.shuffle(buffer_size=len(labels)).batch(self.config.batch_size)

        num_users = len(user_ids)
        num_items = len(poi_ids)
        model = self._build_model(num_users, num_items)
        model.fit(dataset, epochs=self.config.epochs, verbose=1)

        user_embeddings = model.get_layer("user_embedding").get_weights()[0]
        item_embeddings = model.get_layer("item_embedding").get_weights()[0]

        np.savez(
            MODEL_FILE,
            user_ids=user_ids,
            poi_ids=poi_ids,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
        )
        print(f"已保存模型到 {MODEL_FILE}")

    def _build_model(self, num_users: int, num_items: int) -> tf.keras.Model:
        user_input = tf.keras.Input(shape=(1,), name="user", dtype=tf.int32)
        item_input = tf.keras.Input(shape=(1,), name="item", dtype=tf.int32)
        user_vec = tf.keras.layers.Embedding(num_users, self.config.embedding_dim, name="user_embedding")(user_input)
        item_vec = tf.keras.layers.Embedding(num_items, self.config.embedding_dim, name="item_embedding")(item_input)
        dot = tf.keras.layers.Dot(axes=-1)([user_vec, item_vec])
        dot = tf.keras.layers.Flatten()(dot)
        output = tf.keras.layers.Activation("sigmoid")(dot)
        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(self.config.learning_rate), loss="binary_crossentropy")
        return model


class UserCFRecommender:
    def __init__(self, model_file: Path = MODEL_FILE):
        if not model_file.exists():
            raise FileNotFoundError(f"未找到模型文件：{model_file}")
        payload = np.load(model_file, allow_pickle=True)
        self.user_ids = payload["user_ids"]
        self.poi_ids = payload["poi_ids"]
        self.user_embeddings = payload["user_embeddings"]
        self.item_embeddings = payload["item_embeddings"]
        self.user_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        if user_id not in self.user_index:
            return []
        user_vec = self.user_embeddings[self.user_index[user_id]]
        scores = np.dot(self.item_embeddings, user_vec)
        top_indices = np.argpartition(-scores, range(min(top_k, len(scores))))[:top_k]
        ranked = sorted(top_indices, key=lambda idx: scores[idx], reverse=True)
        return [int(self.poi_ids[idx]) for idx in ranked]

