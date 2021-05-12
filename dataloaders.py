from torch.utils.data import DataLoader
from datasets import cats_dogs_dataset


def dataloaders(load_test_data=False):
    training_data = cats_dogs_dataset()
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    if load_test_data:
        pass
    return train_dataloader
