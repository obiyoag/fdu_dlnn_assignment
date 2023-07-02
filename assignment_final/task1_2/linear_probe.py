import torch

from resnet import resnet18
from cifar_trainer import Trainer


class LinearProber(Trainer):
    def __init__(self, exp_name, checkpoint_path):
        super().__init__(exp_name)
        self.model = resnet18()
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        new_state_dict = {}
        for k, v in state_dict.items():
            if 'classifier' not in k:
                new_state_dict[k] = v
        
        new_state_dict['classifier.weight'] = self.model.state_dict()['classifier.weight']
        new_state_dict['classifier.bias'] = self.model.state_dict()['classifier.bias']

        self.model.load_state_dict(new_state_dict)

        self.model.eval().to(self.device)

        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(self.epochs * len(self.train_loader)))


if __name__ == '__main__':
    trainer = LinearProber(exp_name='linear_probe', checkpoint_path='./logs/simclr/simclr.pth')
    trainer.train()
