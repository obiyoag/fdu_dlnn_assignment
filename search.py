from train import Trainer

lrs = [1e-3, 1e-4, 1e-5]
hidden_dims = [128, 256, 512]
weight_decays = [1e-3, 1e-4, 1e-5]

best_acc = 0

for lr in lrs:
    for hidden_dim in hidden_dims:
        for weight_decay in weight_decays:
            trainer = Trainer(lr=lr, hidden_dim=hidden_dim, weight_decay=weight_decay)
            trainer.train()
            if trainer.best_acc > best_acc:
                best_acc = trainer.best_acc
                best_setting = trainer.setting

print(f'The best setting is {best_setting}, with acc {best_acc}')
