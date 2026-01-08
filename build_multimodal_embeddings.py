from __future__ import annotations

from django.core.management.base import BaseCommand

from recommend.recommender.multimodal import MultimodalEmbeddingBuilder


class Command(BaseCommand):
    help = "构建多模态 POI Embedding（图片+文本+标签）"

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=None, help="可选，限制最大处理的景点数量")
        parser.add_argument(
            "--hf_model_path",
            type=str,
            default="recommend/models/clip-vit-base-patch16/clip-vit-base-patch16",
            help="Hugging Face 模型本地路径（包含 pytorch_model.bin 等文件）",
        )

    def handle(self, *args, **options):
        builder = MultimodalEmbeddingBuilder(
            hf_model_path=options["hf_model_path"],
            limit=options["limit"],
        )
        builder.build()
        self.stdout.write(self.style.SUCCESS("多模态 Embedding 构建完成"))

