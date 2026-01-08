from __future__ import annotations

from django.core.management.base import BaseCommand

from recommend.recommender.dnn_rank import DNNRankConfig, DNNRankTrainer


class Command(BaseCommand):
    help = "训练 DNN CTR 排序模型（User + POI Embedding + 多特征全连接）"

    def add_arguments(self, parser):
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--embedding", type=int, default=32)

    def handle(self, *args, **options):
        cfg = DNNRankConfig(
            embedding_dim=options["embedding"],
            epochs=options["epochs"],
        )
        trainer = DNNRankTrainer(config=cfg)
        trainer.train()
        self.stdout.write(self.style.SUCCESS("DNN 排序模型训练完成"))


