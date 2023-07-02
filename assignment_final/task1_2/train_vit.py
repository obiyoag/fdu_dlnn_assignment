import torch
from timm.models import VisionTransformer

from cifar_trainer import Trainer

class VitTrainer(Trainer):
    def __init__(self, exp_name):
        super().__init__(exp_name)
        self.model = VisionTransformer(32, 4, num_classes=100, embed_dim=512, depth=6, num_heads=8).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(self.epochs * len(self.train_loader)))


if __name__ == '__main__':
    trainer = VitTrainer(exp_name='vit')
    trainer.train()
