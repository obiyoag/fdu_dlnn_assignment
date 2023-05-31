from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import VOCDetection
from utils import AnnotationTransform



def get_loaders():
    train_dataset = VOCDetection(root='./dataset',
                           year='2007',
                           image_set='train',
                           download=True,
                           transform=ToTensor(),
                           target_transform=AnnotationTransform())

    test_dataset = VOCDetection(root='./dataset',
                           year='2007',
                           image_set='train',
                           download=True,
                           transform=ToTensor(),
                           target_transform=AnnotationTransform())
    
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, test_loader
