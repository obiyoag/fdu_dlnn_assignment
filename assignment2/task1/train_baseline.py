from cifar100_trainer import Trainer


if __name__ == '__main__':
    trainer = Trainer(exp_name='baseline')
    trainer.train()
