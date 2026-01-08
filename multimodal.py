from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import requests
import tensorflow as tf  # type: ignore
from django.conf import settings
from pyhive import hive
from PIL import Image
from transformers import CLIPProcessor, TFCLIPModel  # type: ignore

MODEL_DIR = Path(settings.BASE_DIR) / "recommend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MULTIMODAL_FILE = MODEL_DIR / "poi_multimodal_embeddings.npz"


class HivePOILoader:
    """从 Hive dwd_attractions 表中读取必要字段。"""

    def __init__(self, limit: int | None = None):
        cfg = settings.HIVE_CONFIG
        self.connection = hive.Connection(
            host=cfg["host"],
            port=cfg["port"],
            username=cfg.get("username"),
            password=cfg.get("password") or None,
            database=cfg.get("database", "default"),
        )
        self.table = cfg.get("table", "dwd_attractions")
        self.limit = limit

    def fetch(self) -> List[Dict[str, object]]:
        limit_clause = f"LIMIT {int(self.limit)}" if self.limit else ""
        sql = f"""
            SELECT poiId, poiName, shortFeatures, tagNameList, coverImageUrl, detailUrl, districtName, heatScore, commentScore
            FROM {self.table}
            WHERE coverImageUrl IS NOT NULL AND coverImageUrl != ''
            {limit_clause}
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
        finally:
            cursor.close()
            self.connection.close()

        results: List[Dict[str, object]] = []
        for row in rows:
            record = dict(zip(columns, row))
            record["poiId"] = int(record["poiId"])
            results.append(record)
        return results


class TagEncoder:
    """将标签列表映射为固定长度的 hashed 向量。"""

    def __init__(self, bucket_size: int = 512):
        self.bucket_size = bucket_size

    def encode(self, tags: Sequence[str] | None) -> tf.Tensor:
        tokens = list(tags) if tags else ["unknown"]
        t = tf.constant(tokens)
        hashed = tf.strings.to_hash_bucket_fast(t, self.bucket_size)
        one_hot = tf.reduce_sum(tf.one_hot(hashed, depth=self.bucket_size), axis=0)
        norm = tf.math.l2_normalize(one_hot[tf.newaxis, :], axis=-1)
        return norm


class MultimodalEmbeddingBuilder:
    """使用 CLIP 生成图片、文本、标签多模态 embedding。"""

    def __init__(
        self,
        hf_model_path: str = "recommend/models/clip-vit-base-patch16/clip-vit-base-patch16",
        limit: int | None = None,
    ):
        self.processor = CLIPProcessor.from_pretrained(hf_model_path)
        self.model = TFCLIPModel.from_pretrained(hf_model_path, from_pt=True)
        self.loader = HivePOILoader(limit=limit)
        self.tag_encoder = TagEncoder()

    def _load_image(self, url: str) -> Image.Image | None:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
            return image
        except Exception:
            return None

    def _text_of(self, record: Dict[str, object]) -> str:
        name = record.get("poiName") or ""
        features = ""
        if record.get("shortFeatures"):
            try:
                lst = json.loads(record["shortFeatures"])
                if isinstance(lst, list):
                    features = " ".join(lst)
            except Exception:
                features = str(record["shortFeatures"])
        tags = ""
        if record.get("tagNameList"):
            try:
                lst = json.loads(record["tagNameList"])
                if isinstance(lst, list):
                    tags = " ".join(lst)
            except Exception:
                tags = str(record["tagNameList"])
        return f"{name}. {features}. {tags}".strip()

    def build(self) -> None:
        pois = self.loader.fetch()
        poi_ids: List[int] = []
        embeddings: List[np.ndarray] = []
        for record in pois:
            image = None
            if record.get("coverImageUrl"):
                image = self._load_image(record["coverImageUrl"])
            tag_list = None
            if record.get("tagNameList"):
                try:
                    tag_list = json.loads(record["tagNameList"])
                except Exception:
                    tag_list = [record["tagNameList"]]
            text = self._text_of(record)
            text_inputs = self.processor(
                text=[text],
                return_tensors="tf",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
            text_emb = self.model.get_text_features(**text_inputs)
            if image is not None:
                image_inputs = self.processor(images=image, return_tensors="tf")
                image_emb = self.model.get_image_features(**image_inputs)
            else:
                image_emb = tf.zeros_like(text_emb)
            text_emb = tf.math.l2_normalize(text_emb, axis=-1)
            image_emb = tf.math.l2_normalize(image_emb, axis=-1)
            tag_emb = self.tag_encoder.encode(tag_list)
            concat = tf.concat([image_emb, text_emb, tag_emb], axis=-1)
            concat = tf.math.l2_normalize(concat, axis=-1)
            embeddings.append(concat.numpy()[0])
            poi_ids.append(int(record["poiId"]))
        np.savez(MULTIMODAL_FILE, poi_ids=np.array(poi_ids, dtype=np.int64), embeddings=np.array(embeddings, dtype=np.float32))
        print(f"已生成 {len(poi_ids)} 条多模态 embedding 并保存到 {MULTIMODAL_FILE}")


class MultimodalRecommender:
    """基于多模态 embedding 的相似景点检索。"""

    def __init__(self, model_file: Path = MULTIMODAL_FILE):
        if not model_file.exists():
            raise FileNotFoundError("未找到多模态模型文件，请先运行 build_multimodal_embeddings 命令")
        payload = np.load(model_file, allow_pickle=True)
        self.poi_ids = payload["poi_ids"]
        self.embeddings = payload["embeddings"]
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        self.embeddings = self.embeddings / norms
        self.id_to_index = {int(poi): idx for idx, poi in enumerate(self.poi_ids)}

    def recommend(self, poi_id: int, top_k: int = 10) -> List[int]:
        if poi_id not in self.id_to_index:
            return []
        base_vec = self.embeddings[self.id_to_index[poi_id]]
        sims = np.dot(self.embeddings, base_vec)
        top_indices = np.argpartition(-sims, range(min(top_k + 1, len(sims))))[: top_k + 1]
        ranked = sorted(top_indices, key=lambda idx: sims[idx], reverse=True)
        result: List[int] = []
        for idx in ranked:
            candidate = int(self.poi_ids[idx])
            if candidate == poi_id:
                continue
            result.append(candidate)
            if len(result) >= top_k:
                break
        return result

