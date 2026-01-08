from __future__ import annotations

from django.core.management.base import BaseCommand

from recommend.recommender.user_cf import TrainingConfig, UserCFTrainer


class Command(BaseCommand):
    help = "训练用户协同过滤(UserCF)模型"

    def add_arguments(self, parser):
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--embedding", type=int, default=32)

    def handle(self, *args, **options):
        config = TrainingConfig(embedding_dim=options["embedding"], epochs=options["epochs"])
        trainer = UserCFTrainer(config=config)
        trainer.train()
        self.stdout.write(self.style.SUCCESS("UserCF 模型训练完成"))

